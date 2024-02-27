from typing import Any

import pytorch_lightning as pl
import torchmetrics
from torch import (
    Tensor,
    all,
    any,
    argmax,
    autograd,
    bool as t_bool,
    cat,
    cuda,
    device,
    distributions,
    div,
    float32,
    log,
    masked_select,
    mean,
    optim,
    pow,
    randn,
    split,
    sum,
    zeros,
)
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, normalize
from torch.optim import AdamW
from transformers import AutoConfig
from transformers.optimization import get_constant_schedule_with_warmup

from models.backbones.adversarial_bert import AdversarialBertModel
from models.discriminator import Discriminator
from models.generator import Generator


class AdvGanBertModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_path: str = "bert-base-uncased",
        num_labels: int = 2,
        num_hidden_layers_disc: int = 1,
        num_hidden_layers_gen: int = 1,
        noise_size: int = 100,
        out_dropout_rate: float = 0.2,
        learning_rate_discriminator: float = 5e-5,
        learning_rate_generator: float = 5e-5,
        use_generator: bool = True,
        adv_penalty: bool = False,
        vadv_penalty: bool = False,
        pi: int = 1,
        adv_epsilon: float = 0.02,
        vadv_epsilon: float = 0.02,
        adv_coef: int = 1,
        vadv_coef: int = 1,
        max_seq_length: int = 512,
    ) -> None:
        """
        A torch lightning module combining the AdvBertModel, the Generator
        network and the Discriminator network. It contains the logic and
        necessary hepler functions for training the network.

        ---parameters---
        pretrained_model_path: a string indicating which model to
        initalize the AdvBert model from among the ones on the hugging faces
        repsoitory.

        num_labels: An integer reprsenting the number of labels for initalizing
        the Discriminator network.

        num_hidden_layers_disc: An integer reprsenting the number of hidden
        layers to have in the Disciminator network.

        num_hidden_layers_gen: An integer reprsenting the number of hidden
        layers to have in the Generator network.

        noise_size: An integer reprsenting the size of the input tensor passed
        to the Generator network.

        out_dropout_rate: A float value reperesenting the dropout rate that
        will be applied to all intermediate layers of the Generator and
        Discriminator networks during training.

        learning_rate_discriminator: A float value representing the learning
        rate in the Discriminator network's optimizer.

        learning_rate_generator: A float value representing the learning
        rate in the Generator network's optimizer.

        use_generator: A Boolelan hyperparameter indicating if the Generator
        network should be used for semi-supervised learning during training

        adv_penalty: A Boolean hyperparameter indicating if the adversarial.
        training regularizer should be applied to the loss.

        vadv_penalty: A Boolean hyperparameter indicating if the generative
        adversarial training regularizer should be applied to the loss.

        pi: An integer hyperparameter indicating how many power iterations
        should be applied when estimating the perturbation for the generative
        adversarial training regularizer.

        adv_epsilon: A float hyperparameter representing the bound placed on
        the perturbation applied for the adversarial training regularizer.

        vadv_epsilon: A float hyperparameter representing the bound placed on
        the perturbation applied for the generatiev adversarial training
        regularizer.

        adv_coef: A float hyperparameter representing the coefficient with
        which the adversarial training penalty should be added to the
        Discriminator network's loss

        vadv_coef: A float hyperparameter representing the coefficient with
        which the generative adversarial training penalty should be added to
        the Discriminator network's loss

        max_seq_length: An integer parameter representing the maximum length
        of an input sequence.

        """

        super().__init__()

        # load the AdvBert model from the hugging faces repository
        self.model = AdversarialBertModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path
        )

        # load the config relating to the loaded model
        config = AutoConfig.from_pretrained(pretrained_model_path)
        # this value represents the size of the tensor produced from the model
        # pooling layer, and is thus also the size outputted by the Generator and
        # the size the Discriminator network expects as input
        self.hidden_size = int(config.hidden_size)
        if not max_seq_length <= int(config.max_position_embeddings):
            raise Exception("Maximum sequence length is longer than config embedding length")
        self.max_seq_length = max_seq_length

        # the hidden size is used in each intermediate layer
        hidden_levels_gen = (self.hidden_size,) * num_hidden_layers_gen
        hidden_levels_disc = (self.hidden_size,) * num_hidden_layers_disc

        # init the discriminator
        self.discriminator = Discriminator(
            input_size=self.hidden_size,
            hidden_sizes=hidden_levels_disc,
            num_labels=num_labels,
            dropout_rate=out_dropout_rate,
        )

        self.use_generator = use_generator
        if self.use_generator:
            # init the generator
            self.generator = Generator(
                noise_size=noise_size,
                output_size=self.hidden_size,
                hidden_sizes=hidden_levels_gen,
                dropout_rate=out_dropout_rate,
            )

        self.learning_rate_discriminator = learning_rate_discriminator
        self.learning_rate_generator = learning_rate_generator

        self.adv_penalty = adv_penalty
        self.adv_coef = adv_coef
        self.adv_epsilon = adv_epsilon

        self.vadv_penalty = vadv_penalty
        self.vadv_coef = vadv_coef
        self.vadv_epsilon = vadv_epsilon
        self.pi = pi

        if adv_penalty and adv_coef is None:
            raise Exception("If using an adversarial penalty you must provide a weight coefficient")
        if adv_penalty and adv_epsilon is None:
            raise Exception(
                "If using an adversarial penalty you must provide a normalization bound"
            )
        if vadv_penalty and vadv_coef is None:
            raise Exception(
                "If using a generative adversarial penalty you must provide a weight coefficient"
            )
        if vadv_penalty and vadv_epsilon is None:
            raise Exception(
                "If using a generative adversarial penalty you must provide a normalization bound"
            )
        if vadv_penalty and pi is None:
            raise Exception(
                "If using a generative adversarial penalty you must provide a power iteration value"
            )

        # activated manual optimization telling torch lightning that the logic
        # for the optimizer step will be included in the training step
        # this is necessary to successfully apply the GAN game
        self.automatic_optimization = False

        # Accuracy tracking for logs
        if num_labels > 2:
            self.train_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.discriminator.num_labels
            )
            self.val_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.discriminator.num_labels
            )
            self.test_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.discriminator.num_labels
            )
            self.train_f1_Score = torchmetrics.classification.MulticlassF1Score(
                num_classes=self.discriminator.num_labels
            )
            self.test_f1_Score = torchmetrics.classification.MulticlassF1Score(
                num_classes=self.discriminator.num_labels
            )
            self.val_f1_Score = torchmetrics.classification.MulticlassF1Score(
                num_classes=self.discriminator.num_labels
            )

        elif num_labels == 2:
            self.train_acc = torchmetrics.Accuracy(
                task="binary", num_classes=self.discriminator.num_labels
            )
            self.val_acc = torchmetrics.Accuracy(
                task="binary", num_classes=self.discriminator.num_labels
            )
            self.test_acc = torchmetrics.Accuracy(
                task="binary", num_classes=self.discriminator.num_labels
            )
            self.train_f1_Score = torchmetrics.classification.BinaryF1Score()
            self.test_f1_Score = torchmetrics.classification.BinaryF1Score()
            self.val_f1_Score = torchmetrics.classification.BinaryF1Score()

        # logging hyperparameters
        self.save_hyperparameters()

    @staticmethod
    def compute_generator_loss(
        generator_probs: Tensor,
        discriminator_representation: Tensor,
        generator_representation: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Calculate the loss to be used by the Generator during backpropagation.
        It penalizes the Generator proportional to how certain the Discriminator
        was in classifying a Generated input as produced by the Generator.
        It adds to this a loss that penalizes the Generator proportional to how
        "far" its output was compared to the last hidden layer of the
        Discriminator network. For more information refer to arXiv:1606.03498.

        ---parameters---
        generator_probs: A tensor containing the log probabilities assigned
        across each of the num_labels+1 classes by the Discriminator network
        for an input produced by the Generator netowrk.

        discriminator_representation: A tensor containing the output assigned
        by the last hidden layer of the Discriminator network
        for an input produced by the Generator netowrk.

        generator_representation: A tensor containing the output of the Generator
        network for a random noise input.

        ---output---
        A floating point value representing the average loss for the Generator
        network on a batch.
        """
        g_loss_d = -1 * mean(log(1 - generator_probs[:, -1] + eps))  # only on the k+1 fake class
        g_feat_reg = mean(
            pow(
                mean(discriminator_representation, dim=0) - mean(generator_representation, dim=0),
                2,
            )
        )  # feature matching loss
        g_loss = g_loss_d + g_feat_reg

        return g_loss

    def compute_adversarial_loss(
        self,
        embedding_gradients: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        labels: Tensor,
        label_mask: Tensor,
        epsilon: float = 0.02,
    ) -> Tensor:
        """
        Implemets the adversarial loss calculation for a given batch.
        The function works by performing a second forward pass through the
        model with the same batch where the output of the Bert model's
        embedding layer is now perturbed using the gradient of the masked
        cross entropy loss with respect to the unperturbed embedding layer
        output. Finally masked cross entropy is calculated on this output.

        ---parameters---
        embedding_gradient: A tensor of shape [batch_size, hidden_size, 768]
        containing the gradeint of the current batch's masked cross entropy
        loss with respect to the output of the Bert embedding layer.

        input_ids: A tensor of shape [batch_size, hidden_size] containing
        the tokenized input. An input to the Bert model.

        attention_mask: A tensor of shape [batch_size, hidden_size] indicating
        the tokens to attend to. An input to the Bert model.

        token_type_ids: A tensor of shape [batch_size, hidden_size] indicating
        the type of each token. An input to the Bert model.

        labels: A tensor of shape [batch_size] containing the
        labels per input.

        label_mask: A tensor of shape [batch_size] indicating which examples
        to consider unlabelled.

        epsilon: A float hyperparameter representing the bound placed on
        the perturbation applied for the adversarial training regularizer.

        ---output---
        The average cross entropy loss on the adversarially perturbed batch.

        """

        # l2 normalize the perturbation for each input across the embedding dimension
        # the expected output should return a tensor of shape [batch_size, 512, 768]
        # here the hidden_size=512 is the padded input vector size (max number of tokenes) per input and 768 is the dimension of the output emedding space
        # a successful normalization will modify the perturbation such that the sum of squares for each embedding dimension for a given input is 1

        l2_norm = normalize(embedding_gradients, dim=1)

        perturbation = epsilon * l2_norm  # epsilon model hyperparameter

        # forward pass with perturbation
        _, perturbed_logits, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            perturbation=perturbation,
            use_generator=False,
            output_embedding_layer=False,
        )

        # masked cross entropy
        loss = cross_entropy(perturbed_logits[:, 0:-1], labels)

        return loss

    def compute_virtual_adversarial_loss(
        self,
        real_logits: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        pi: int = 1,
        epsilon: float = 0.02,
    ) -> Tensor:
        """
        Implemets the generative adversarial loss calculation for a given batch.
        The function works by performing a possibly several forward passes
        through the model. In the first iteration the current batch is
        perturbed according to a normalized random tensor. Subsequent
        perturbations are calculated as the gradient of the Kullback-Leibler
        divergence between the unperturbed Bert pooled output and the most recent
        perturbed Bert pooled output. The finall Kullback-Leibler divergence
        representes the final loss.

        ---parameters---
        real_logits: A tensor of shape [batch_size, hidden_size] containing
        the output of the Bert pooling layer for the current batch.

        input_ids: A tensor of shape [batch_size, hidden_size] containing
        the tokenized input. An input to the Bert model.

        attention_mask: A tensor of shape [batch_size, hidden_size] indicating
        the tokens to attend to. An input to the Bert model.

        token_type_ids: A tensor of shape [batch_size, hidden_size] indicating
        the type of each token. An input to the Bert model.

        labels: A tensor of shape [batch_size] containing the
        labels per input.

        pi: An interger hyperpatameter representing the number of times to
        perform the power iteration approximation.

        epsilon: A float hyperparameter representing the bound placed on
        the perturbation applied for the adversarial training regularizer.

        ---output---
        The kullback-leibler divergence between the unperturbed Bert pooler
        output and the Generative Adversarially perturbed Bert pooler output for
        the current batch.

        """
        # normalized random noise vector
        noise = randn(size=(real_logits.shape[0], self.max_seq_length, self.hidden_size))
        l2_norm = normalize(noise, dim=1)

        for _ in range(pi):  # power iteration method
            perturbation = epsilon * l2_norm  # epsilon model hyperparameter
            # forward pass
            _, perturbed_logits, _, embedding_output = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                perturbation=perturbation,
                use_generator=False,
                output_embedding_layer=True,
            )
            # KL divergence
            loss = self.kl_divergence(perturbed_logits[:, 0:-1], real_logits)
            # Gradient of KL divergence wrt embedding output
            embedding_gradients = autograd.grad(
                outputs=loss,
                inputs=embedding_output,
                retain_graph=True,
                create_graph=True,
            )[0]
            # normalize
            l2_norm = normalize(embedding_gradients, dim=1)

        perturbation = epsilon * l2_norm  # epsilon model hyperparameter

        # final forward pass
        _, perturbed_logits, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            perturbation=perturbation,
            use_generator=False,
            output_embedding_layer=False,
        )

        # final loss
        loss = self.kl_divergence(perturbed_logits[:, 0:-1], real_logits)

        return loss

    @staticmethod  # TODO: make helper
    def kl_divergence(logits_perturbed: Tensor, logits_unperturbed: Tensor) -> Tensor:
        """
        Implements the kullback-leibler divergence between two sets of logits.

        ---parameters---
        logits_perturbed: A tensor of shape [batch_size, hidden_size]
        containing the logits produced by the Bert pooler for a batch
        where the embedding output has been perturbed.

        logits_unperturbed: A tensor of shape [batch_size, hidden_size]
        containing the logits produced by the Bert pooler for a batch
        where the embedding output has not been perturbed.

        ---output---
        A float value representing the average KL divergence for each
        set of logits in the batch.
        """

        p = Categorical(logits=logits_perturbed)

        q = Categorical(logits=logits_unperturbed)

        return mean(distributions.kl.kl_divergence(p, q))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        use_generator: bool = False,
        perturbation: Tensor | None = None,
        output_embedding_layer: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Implements the forward pass of the model. This corresponds to a
        forward pass through the AdvBert and Discriminator networks.
        According to the hyperparameters it may also utilize the Generator
        network to augment the data.

        ---parameters---
        input_ids: A tensor of shape [batch_size, hidden_size] containing
        the tokenized input. An input to the Bert model.

        attention_mask: A tensor of shape [batch_size, hidden_size] indicating
        the tokens to attend to. An input to the Bert model.

        token_type_ids: A tensor of shape [batch_size, hidden_size] indicating
        the type of each token. An input to the Bert model.

        use_generator: A boolean value indicating if the Generator network
        should be used to augment the batch size.

        perturbation: Either None or a tensor of shape [batch_size, hidden_size, 768]
        containing the perturbation to be applied to the Bert embedding output.

        output_embedding_layer: A boolean value indicating if the output of the
        Bert embedding layer should be returned by the function.

        ---output---
        The output of the last hidden layer of the Discriminator network,
        the output of the final linear layer of the Discriminator network and
        the output of a softmax layer applied on the final linear layer of the
        Discriminator network. Will also return the output of the Bert
        embedding layer if specified.
        """
        # Forward pass through model model
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            perturbation=perturbation,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # Bert pooler output
        hidden_states = model_outputs.get("pooler_output")

        if use_generator:
            # Generate fake data that should have the same distribution of the ones encoded by the transformer.
            # First noisy input are used in input to the Generator
            noise = zeros(
                input_ids.shape[0], self.generator.noise_size, device=self.device
            ).uniform_(
                0, 1
            )  # TODO check device
            # Gnerate Fake data
            gen_rep = self.generator(noise)

            # Generate the output of the Discriminator for real and fake data.
            # First, we put together the output of the transformer and the generator
            disciminator_input = cat([hidden_states, gen_rep], dim=0)

            # Then, we select the output of the disciminator
            last_representation, logits, probs = self.discriminator(disciminator_input)
            if output_embedding_layer:
                return (
                    last_representation,
                    logits,
                    probs,
                    model_outputs.get("hidden_states")[0],
                )  # output of model embedding layer
            else:
                return last_representation, logits, probs

        else:
            last_representation, logits, probs = self.discriminator(hidden_states)
            if output_embedding_layer:
                return (
                    last_representation,
                    logits,
                    probs,
                    model_outputs.get("hidden_states")[0],
                )  # output of model embedding layer
            else:
                return last_representation, logits, probs

    def configure_optimizers(
        self,
    ) -> tuple[list[AdamW], list[Any]] | dict[str, AdamW | Any]:
        """
        Configures the optimizers for the Disciminator and Generator networks
        according to the lelarning rates provided on model initalilzation.

        ---output---
        The optimizer for the Generator and Discriminator networks.
        """
        # models parameters
        transformer_vars = [i for i in self.model.parameters()]
        d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
        # optimizer
        dis_optimizer = optim.AdamW(d_vars, lr=self.learning_rate_discriminator)
        dis_scheduler = get_constant_schedule_with_warmup(
            optimizer=dis_optimizer, num_warmup_steps=0
        )
        if self.use_generator:
            # model parameters
            g_vars = [v for v in self.generator.parameters()]
            # optimizer
            gen_optimizer = optim.AdamW(g_vars, lr=self.learning_rate_generator)
            gen_scheduler = get_constant_schedule_with_warmup(
                optimizer=gen_optimizer, num_warmup_steps=0
            )
            return [dis_optimizer, gen_optimizer], [dis_scheduler, gen_scheduler]

        return {"optimizer": dis_optimizer, "lr_scheduler": dis_scheduler}

    # todo: fix any typing
    def training_step(self, batch: Any, batch_idx: int) -> None:
        """
        Implements the logic for the training loop of the model. Note that
        because the model is using manual optimization all training logic
        is dictated by this function.

        ---parameters---
        batch: A single batch from the dataloader.
        batch_idx: The index of the batch
        """
        # Avoid gradient accumulation
        if self.use_generator:
            dis_optimizer, gen_optimizer = self.optimizers()[0], self.optimizers()[1]
            dis_scheduler, gen_scheduler = self.lr_schedulers()[0], self.lr_schedulers()[1]
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()
        else:
            dis_optimizer = self.optimizers()
            dis_scheduler = self.lr_schedulers()
            dis_optimizer.zero_grad()

        # init variables
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        token_type_ids = batch.get("token_type_ids")
        labels = batch.get("labels")
        label_mask = batch.get("label_mask").to(t_bool)  # torch bool

        # batch size without generator
        real_batch_size = input_ids.shape[0]

        # forward pass on the real data + the generator if specified
        features, logits, probs, embedding_outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_generator=self.use_generator,
            output_embedding_layer=True,
        )
        # If we are training with a generator...
        if self.use_generator:
            # We separate the discriminator's output for the real and fake (generator) data
            features_list = split(
                features, real_batch_size
            )  # split tensor into even chunks of size real_batch_size
            D_real_features = features_list[0]  # first index is real feature due to concat order
            D_fake_features = features_list[1]  # fake refers to outputs on the generated data

            logits_list = split(logits, real_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]

            probs_list = split(probs, real_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

            # use the helper function to calculate the generator loss
            g_loss = self.compute_generator_loss(
                generator_probs=D_fake_probs,
                discriminator_representation=D_real_features,
                generator_representation=D_fake_features,
                eps=1e-8,
            )

            # Calculate the unsupervised discriminator loss based on its
            # Certainty/uncertainty for real/fake data
            D_L_unsupervised1U = -1 * mean(log(1 - D_real_probs[:, -1] + 1e-8))
            D_L_unsupervised2U = -1 * mean(log(D_fake_probs[:, -1] + 1e-8))

            # Disciminator's LOSS estimation IMPLEMENTS NEGATIVE LOG LIKELIHOOD / CROSS ENTROPY LOSS
            # No need to run it if the label contains no labeled data
            if any(label_mask == True):
                D_L_Supervised = cross_entropy(
                    D_real_logits[label_mask, 0:-1],
                    labels[label_mask],
                )
            else:
                D_L_Supervised = zeros((), device=self.device)

            # no need to compute adv loss if no labels in batch
            if all(D_L_Supervised) != 0 and self.adv_penalty:
                if not isinstance(D_L_Supervised, Tensor):
                    raise Exception("Supervised loss object must be a Tensor")
                # gradients wrt embedding output
                embedding_gradients = autograd.grad(
                    outputs=D_L_Supervised,
                    inputs=embedding_outputs,  # output of model embedding layer for labeled examples
                    retain_graph=True,
                    create_graph=True,
                )[0].detach()
                # calculate adversarial loss
                adversarial_penalty = self.compute_adversarial_loss(
                    embedding_gradients=embedding_gradients[label_mask],
                    input_ids=input_ids[label_mask],
                    attention_mask=attention_mask[label_mask],
                    token_type_ids=token_type_ids[label_mask],
                    labels=labels[label_mask],
                    label_mask=label_mask[label_mask],
                    epsilon=self.adv_epsilon,
                )
            else:
                adversarial_penalty = zeros((), device=self.device)

            # compute vadv loss
            if self.vadv_penalty and any(label_mask == False):
                vadv_penalty = self.compute_virtual_adversarial_loss(
                    real_logits=D_real_logits[~label_mask, 0:-1],
                    input_ids=input_ids[~label_mask],
                    attention_mask=attention_mask[~label_mask],
                    token_type_ids=token_type_ids[~label_mask],
                    pi=self.pi,
                    epsilon=self.vadv_epsilon,
                )
            else:
                vadv_penalty = zeros((), device=self.device)

        # If no gennerator same steps as above except discriminator's unsupervised
        # losses are zero and there are no generated data or genenrator loss
        else:
            D_L_unsupervised1U = zeros((), device=self.device)
            D_L_unsupervised2U = zeros((), device=self.device)
            # Disciminator's LOSS estimation IMPLEMENTS NEGATIVE LOG LIKELIHOOD / CROSS ENTROPY LOSS
            if any(label_mask == True):
                D_L_Supervised = cross_entropy(
                    logits[label_mask, 0:-1],
                    labels[label_mask],
                )
            else:
                D_L_Supervised = zeros((), device=self.device)

            if all(D_L_Supervised) != 0 and self.adv_penalty:
                embedding_gradients = autograd.grad(
                    outputs=D_L_Supervised,
                    inputs=embedding_outputs,  # output of model embedding layer for labeled examples
                    retain_graph=True,
                    create_graph=True,
                )[0].detach()

                adversarial_penalty = self.compute_adversarial_loss(
                    embedding_gradients=embedding_gradients[label_mask],
                    input_ids=input_ids[label_mask],
                    attention_mask=attention_mask[label_mask],
                    token_type_ids=token_type_ids[label_mask],
                    labels=labels[label_mask],
                    label_mask=label_mask[label_mask],
                    epsilon=self.adv_epsilon,
                )
            else:
                adversarial_penalty = zeros((), device=self.device)

            if self.vadv_penalty and any(label_mask == False):
                vadv_penalty = self.compute_virtual_adversarial_loss(
                    real_logits=logits[~label_mask, 0:-1],
                    input_ids=input_ids[~label_mask],
                    attention_mask=attention_mask[~label_mask],
                    token_type_ids=token_type_ids[~label_mask],
                    pi=self.pi,
                    epsilon=self.vadv_epsilon,
                )
            else:
                vadv_penalty = zeros((), device=self.device)

        # discriminator loss as sum of the multiple components
        # adv and vadv loss are multiplied by coef here
        d_loss = (
            D_L_Supervised
            + D_L_unsupervised1U
            + D_L_unsupervised2U
            + (self.adv_coef * adversarial_penalty)
            + (self.vadv_coef * vadv_penalty)
        )

        # retain_graph=True is required since the underlying graph will be deleted after backward
        if self.use_generator:
            self.manual_backward(g_loss, retain_graph=True)
        if d_loss.requires_grad:
            self.manual_backward(d_loss)
            dis_optimizer.step()
            dis_scheduler.step()

        # Apply modifications
        if self.use_generator:
            gen_optimizer.step()
            gen_scheduler.step()

        if self.use_generator:
            logits = D_real_logits

        # calcualte the training accuracy and log it
        if any(label_mask == True):
            self.train_acc(argmax(logits[label_mask, 0:-1], dim=1), labels[label_mask])
            self.train_f1_Score(argmax(logits[label_mask, 0:-1], dim=1), labels[label_mask])
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_f1", self.train_f1_Score, on_step=True, on_epoch=True, prog_bar=True)

        # log the loss of the generator and discriminator
        if self.use_generator:
            self.log("train_loss_gen", g_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_disc", d_loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Implements the logic for a single step in the validation loop.
        """

        # initalize data
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        token_type_ids = batch.get("token_type_ids")
        labels = batch.get("labels")
        label_mask = batch.get("label_mask").to(t_bool)  # torch bool

        # forward pass
        features, logits, probs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            perturbation=None,
            use_generator=False,
            output_embedding_layer=False,
        )

        # calculate val loss as masked cross entropy loss and log it
        if sum(label_mask) != 0:
            val_loss = cross_entropy(logits[label_mask, 0:-1], labels[label_mask])
            self.log(
                "val_loss_disc",
                0 if val_loss is None else val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # calculate accuracy and log it
        self.val_acc(argmax(logits[label_mask, 0:-1], dim=1), labels[label_mask])
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # calculate f1 score and log it
        self.val_f1_Score(argmax(logits[label_mask, 0:-1], dim=1), labels[label_mask])
        self.log("val_f1", self.val_f1_Score, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """
        Implements the logic for a single step in the validation loop.
        """

        # initalize data
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        token_type_ids = batch.get("token_type_ids")
        labels = batch.get("labels")
        label_mask = batch.get("label_mask").to(t_bool)  # torch bool

        # forward pass
        features, logits, probs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            perturbation=None,
            use_generator=False,
            output_embedding_layer=False,
        )

        # calcualte test loss as masked cross entropy loss as log it
        if sum(label_mask) != 0:
            test_loss = cross_entropy(logits[label_mask, 0:-1], labels[label_mask])
            self.log(
                "test_loss",
                test_loss if test_loss is not None else test_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # calculate accuracy and log it
        self.test_acc(argmax(logits[label_mask, 0:-1], dim=1), labels[label_mask])
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        # calculate f1 score and log it
        self.test_f1_Score(argmax(logits[label_mask, 0:-1], dim=1), labels[label_mask])
        self.log("test_f1", self.test_f1_Score, on_step=True, on_epoch=True, prog_bar=True)
