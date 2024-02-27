from torch import Tensor
from torch.nn import Dropout, LeakyReLU, Linear, Module, Sequential, Softmax


class Discriminator(Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_sizes: tuple[int, ...] = (768,),
        num_labels: int = 2,
        relu_slope: float = 0.2,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Initalizer for the discriminator network. It takes as input a tensor of
        size input_size which is either the output of the Bert pooling layer or
        the Generator network and tries to classify it according to num_labels+1
        classes, where the +1 class corresponds to predicting the data was
        produced by the generator.

        ---parameters---
        input_size: integer value representing the size of the tensor produced
        by the Bert pooling layer/Generator network.

        hidden_sizes: An iterable of integer values representing the output
        dimension of each intermediate feed forward layer.
        """
        super(Discriminator, self).__init__()

        self.num_labels = num_labels

        self.input_dropout = Dropout(dropout_rate)

        layers = []

        hidden_sizes = (input_size, *hidden_sizes)  # add input_size as first size

        # Create as many linear layers as in hidden_sizes
        # Apply the RELu activation and dropout to each
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    LeakyReLU(relu_slope, inplace=True),
                    Dropout(dropout_rate),
                ]
            )

        # Init also a softmax layer and final hidden layer
        self.layers = Sequential(*layers)
        self.logit = Linear(
            hidden_sizes[-1], num_labels + 1
        )  # +1 for the probability of this sample being fake or real.
        self.softmax = Softmax(dim=-1)

    def forward(self, input_representation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        The forward call for the discriminator network. Given the input
        it applies the linear layers, a final output linear layer and a
        softmax layer.

        ---parameters---
        input_representation: tensor of size input_size. It is the output
        of either the Bert pooling layer or the Generator network.

        ---output---
        A tuple containing the output of the last hidden layer, the output of
        the final linear layer (output layer) and the result of a softmax layer
        applied on this output layer. All three are used at various stages in
        the training loss calculation.
        """
        input_representation = self.input_dropout(input_representation)
        last_representation = self.layers(input_representation)
        logits = self.logit(last_representation)
        probs = self.softmax(logits)
        return last_representation, logits, probs
