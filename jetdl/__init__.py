from ._C import Tensor
from ._creation import ones, zeros
from ._manip import contiguous, reshape, squeeze, unsqueeze
from .linalg import dot, matmul, matrix_transpose, transpose
from .math import add, div, mul, sub, pow, sum, mean


def tensor(data, *, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad)


__all__ = [
    "tensor",
    "Tensor",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sum",
    "mean",
    "dot",
    "matmul",
    "transpose",
    "matrix_transpose",
    "zeros",
    "ones",
    "contiguous",
    "reshape",
    "squeeze",
    "unsqueeze",
]
