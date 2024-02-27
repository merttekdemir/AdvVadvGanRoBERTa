import gc
import itertools

import wandb
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import cuda
from yaml.loader import SafeLoader

from dataloaders.hausa_newsclass_topic import HausaNewsclassDataModule
from models.adversarial_gan_models.adversarial_gan_xlm_roberta import AdvGanXLMRobertaModel


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, min_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.min_steps = min_steps

    def on_validation_end(self, trainer, pl_module):
        # Check if the current global step is less than min_steps
        if trainer.global_step < self.min_steps:
            return  # Skip the early stopping check

        # Proceed with the normal early stopping logic
        super().on_validation_end(trainer, pl_module)


if __name__ == "__main__":
    gc.collect()
    cuda.empty_cache()
    # dictionary of hyperparams to experiment on
    with open(
        "../conf/experiment_config_hausa_few_shot.yaml",
        "r",
    ) as f:
        data = yaml.load(f, Loader=SafeLoader)

    # create a list of every combination of the hyperparams to perform an
    # exhaustive grid search
    keys, values = zip(*data["hparams"].items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    api_key = input("Enter WandB API key: ")
    wandb.login(key=str(api_key), relogin=True)

    FEW_SHOT_DATAPATH = data["few-shot-datapath"]
    UNSUPERVISED_DATAPATH = data["unsupervised-datapath"]
    TEST_DATAPATH = data["test-datapath"]

    # main experiment loop
    for idx, params in enumerate(permutations_dicts):
        # set the seed
        seed_everything(params["seed"] * 2, workers=True)

        # initalize the model
        model = AdvGanXLMRobertaModel(
            pretrained_model_path=params["pretrained-model-path"],
            num_labels=params["num-labels"],
            num_hidden_layers_disc=params["num-hidden-layers-disc"],
            num_hidden_layers_gen=params["num-hidden-layers-gen"],
            noise_size=params["noise-size"],
            out_dropout_rate=params["out-dropout-rate"],
            learning_rate_discriminator=params["learning-rate-discriminator"],
            learning_rate_generator=params["learning-rate-generator"],
            use_generator=params["use-generator"],
            adv_penalty=params["adv-penalty"],
            vadv_penalty=params["vadv-penalty"],
            pi=params["pi"],
            adv_epsilon=params["adv-epsilon"],
            vadv_epsilon=params["vadv-epsilon"],
            adv_coef=params["adv-coef"],
            vadv_coef=params["vadv-coef"],
            max_seq_length=params["max-seq-length"],
        )

        seed_everything(params["seed"], workers=True)
        if params["vadv-penalty"]:
            downsample_unlabeled = True

        else:
            downsample_unlabeled = False
        # initalize the data module
        dm = HausaNewsclassDataModule(
            batch_size=params["batch-size"],
            num_labeled_examples=params["num-labeled-examples"],
            downsample_unlabeled=downsample_unlabeled,
            upsample_labeled=False,
            max_seq_length=params["max-seq-length"],
        )

        # load the data from disk and prepare it
        dm.load_pickle_data(split="few_shot", path=FEW_SHOT_DATAPATH)
        if params["vadv-penalty"]:
            dm.load_pickle_data(split="unsupervised", path=UNSUPERVISED_DATAPATH)
        dm.load_pickle_data(split="test", path=TEST_DATAPATH)

        dm.prepare_data(split="few_shot")
        dm.prepare_data(split="test")

        dm.setup(stage="few_shot")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # Save the checkpoint for the model at the highest validation accuracy
        # checkpoint_callback = ModelCheckpoint(monitor="val_f1", mode="max")
        early_stop_callback = CustomEarlyStopping(
            min_steps=50, monitor="val_f1", min_delta=0.00, patience=3, verbose=False, mode="max"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=None, save_top_k=0  # Set to None to disable saving to disk
        )  # No checkpoints are saved based on monitored metrics

        # Define the name space to save the model experiment on weights and biases
        name = str(params["seed"]) + "_few_shot"
        if params["use-generator"]:
            name = "Gan_" + name
        if params["vadv-penalty"]:
            name = "Vadv_" + name
        if params["adv-penalty"]:
            name = "Adv_" + name
        project = f"hausa_{params['num-labeled-examples']}_shot"

        # log with weights and biases and also save the checkpoint to gdrive
        logger = WandbLogger(name=name, project=project, log_model=False)

        # creat a torch lightning trainer and fit the model
        trainer = Trainer(
            max_epochs=-1,
            min_steps=50,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator="auto",
            deterministic=True,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # empty the cache for the next experiment
        gc.collect()
        cuda.empty_cache()

        # finsh the logging of this model
        wandb.finish()
        print(f"Finished fitting run number {idx} on params: \n {params}")
