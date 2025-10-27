from .._C import Tensor
from ._objects import Module


class MSELoss(Module):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input `x` and target `y`.

    The loss is calculated as: :math:`\text{MSE}(x, y) = {1 \over N} \sum_{i=1}^N (x_i - y_i)^2`,
    where `N` is the total number of elements in the tensor.
    """

    def __init__(self):
        """Initializes the MSELoss module."""
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculates the mean squared error between input and target.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor, which has the same shape as the input.

        Returns:
            Tensor: A scalar tensor representing the mean squared error.
        """
        squared_diff = (input - target) ** 2
        return squared_diff.mean()
