from typing import TYPE_CHECKING
from ..tensor import Tensor
from .._Cpp import (
    c_dot,
    c_matmul
)

def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot product between two tensors.
    Both tensors must be of ndim == 1 and of same shape.
    Args:
        a (Tensor): First input tensor
        b (Tensor): Second input tensor
    Returns:
        Tensor: Result of dot product between tensors a and b
    Note:
        The shapes of input tensors must be compatible for matrix multiplication.
        Both tensors must be of ndim == 1 and of same shape.
    """
    return c_dot(a, b)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication of two tensors.
    This is an operator that performs matrix multiplication of any two compatible tensors of ndim > 0.
    Args:
            a (Tensor): First tensor to be multiplied
            b (Tensor): Second tensor to be multiplied
    Returns:
            Tensor: The result of the matrix multiplication
    """

    return c_matmul(a, b)