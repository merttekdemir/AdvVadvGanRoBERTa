from torch import Tensor
from torch.nn import Dropout, LeakyReLU, Linear, Module, Sequential, Softmax


class Generator(Module):
    def __init__(
        self,
        noise_size: int = 100,
        output_size: int = 768,
        hidden_sizes: tuple[int, ...] = (768,),
        relu_slope: float = 0.2,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Initalilzer for the Generator model. It is a feed forward network taking
        as input a random noise vector and producing as output a tensor trained
        to represent the output of the bert pooling layer.

        ---parameters---
        noise_size: Integer value representing the dimension of the random noise
        vector to be passed as input to the generator

        output_size: Interger value representing the dimension of the output to
        be prodcued from the final linear layer.

        hidden_sizes: An iterable of integer values representing the output
        dimension of each intermediate feed forward layer.

        relu_slope: Float value for the slope coefficient in the relu actviation
        function

        dropout_rate: Float value representing the fraction of neurons to
        deactivate in across each feedforward layer during training

        ---output---
        torch neural network module
        """
        super(Generator, self).__init__()

        self.noise_size = noise_size

        layers = []

        hidden_sizes = (noise_size, *hidden_sizes)  # Append noise_size as first size

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

        # Add the output final linear layer
        layers.append(Linear(hidden_sizes[-1], output_size))

        self.layers = Sequential(*layers)

    def forward(self, noise: Tensor) -> Tensor:
        """
        Forward pass of the model simply takes as input the random noise vector
        and applies the MLP network.

        ---parameters---
        noise: A (gaussian) random tensor of shape noise_size

        ---output---
        A tensor of size output_size
        """
        output_representation = self.layers(noise)
        return output_representation
