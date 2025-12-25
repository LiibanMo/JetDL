from .._C import Tensor
from .._C.nn import c_linear_forward
from ._objects import Module, Parameter


class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    The input and output tensors are of shape `(N, *, in_features)` and
    `(N, *, out_features)` respectively, where `*` means any number of
    additional dimensions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_type: str = "he",
        seed: int = 123,
        device: str = "cpu",
    ) -> None:
        """Initializes the Linear layer.

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            init_type (str, optional): Initialization type for the weight matrix. Defaults to "he".
            seed (int, optional): Random seed for weight initialization. Defaults to 123.
            device (str, optional): The device to create parameters on. Defaults to "cpu".
        """
        super().__init__()

        self.weight = Parameter([out_features, in_features], init_type, seed, device)
        self.bias = Parameter([out_features], "zero", seed, device)

    def forward(self, input: Tensor) -> Tensor:
        """Defines the forward pass of the Linear layer.

        Args:
            input (Tensor): The input tensor of shape `(N, *, in_features)`.

        Returns:
            Tensor: The output tensor of shape `(N, *, out_features)`.
        """
        return c_linear_forward(input, self.weight, self.bias)
