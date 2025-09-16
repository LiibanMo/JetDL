from ._C import Tensor
from ._C.linalg import c_dot, c_matmul, c_mT, c_T


def dot(a: Tensor, b: Tensor) -> Tensor:
    return c_dot(a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return c_matmul(a, b)


def transpose(input: Tensor) -> Tensor:
    return c_T(input)


def matrix_transpose(input: Tensor) -> Tensor:
    return c_mT(input)
