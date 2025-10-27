from .._C import Tensor
from .._C.nn import c_relu_forward
from ._objects import Module


class ReLU(Module):
    r"""Applies the Rectified Linear Unit function element-wise.

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    """

    def __init__(self) -> None:
        """Initializes the ReLU activation function module."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies the ReLU function to the input tensor.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with ReLU applied.
        """
        return c_relu_forward(input)
