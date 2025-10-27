from ._C import Tensor
from ._C.linalg import c_dot, c_matmul, c_mT, c_T


def dot(a: Tensor, b: Tensor) -> Tensor:
    """Computes the dot product of two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.
    """
    return c_dot(a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Computes the matrix product of two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.
    """
    return c_matmul(a, b)


def transpose(input: Tensor) -> Tensor:
    """Returns the transpose of a tensor.

    Args:
        input (Tensor): The input tensor.
    """
    return c_T(input)


def matrix_transpose(input: Tensor) -> Tensor:
    """Returns the matrix transpose of a tensor.

    Args:
        input (Tensor): The input tensor.
    """
    return c_mT(input)


__all__ = [
    "dot",
    "matmul",
    "transpose",
    "matrix_transpose",
]
