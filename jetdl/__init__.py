from ._C import Tensor
from .creation import ones
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
    "ones",
]
