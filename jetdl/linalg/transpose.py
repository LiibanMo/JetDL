from .._Cpp import c_transpose, c_matrix_transpose
from ..tensor import Tensor

def transpose(a: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
    """
    Swaps the first and last dimensions of the input tensor.

    Args:
        a (Tensor): The input tensor.
        axes (tuple[int, ...] | None): Not yet implemented. Custom axes for transpose are not yet implemented.

    Returns:
        Tensor: The tensor with its first and last dimensions swapped.
    """
    return c_transpose(a)

def matrix_transpose(a: Tensor) -> Tensor:
    """
    Transposes a 2D matrix (swaps rows and columns) or the last two dimensions
    of a higher-dimensional tensor.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The matrix-transposed tensor.
    """
    return c_matrix_transpose(a)
