from .._C.linalg import c_T, c_mT
from .._tensor import Tensor

def transpose(input: Tensor) -> Tensor:
    """Computes the full transpose of the input tensor.
    This function reverses the dimensions of the input tensor. 
    Args:
        input (Tensor): The tensor to be transposed.
    Returns:
        Tensor: A new tensor with its dimensions reversed.
    """
    return c_T(input)

def matrix_transpose(input: Tensor) -> Tensor:
    """Computes the transpose of a 2D matrix (or batch of matrices).
    This function swaps the last two dimensions of the input tensor.
    Args:
        input (Tensor): The tensor to be transposed.
    Returns:
        Tensor: A new tensor with its last two dimensions swapped.
    """
    return c_mT(input)