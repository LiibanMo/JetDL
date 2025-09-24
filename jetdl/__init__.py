from ._C import Tensor
from ._creation import ones, zeros
from ._manip import contiguous, reshape, squeeze, unsqueeze
from .linalg import dot, matmul, matrix_transpose, transpose
from .math import add, div, mul, sub, sum


def tensor(data, *, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad)


__all__ = [
    "tensor",
    "Tensor",
    "add",
    "sub",
    "mul",
    "div",
    "sum",
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
