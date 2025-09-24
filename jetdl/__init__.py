from ._C import Tensor
from ._creation import (
    zeros,
    ones,
)
from .linalg import (
    dot,
    matmul,
    matrix_transpose,
    transpose,
)
from ._manip import (
    contiguous,
    squeeze,
    unsqueeze,
    reshape,
)
from .math import (
    add,
    div,
    mul,
    sub,
    sum,
)


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
