from . import linalg, nn, optim
from ._C import Tensor
from ._creation import fill, ones, zeros
from ._manip import contiguous, reshape, squeeze, unsqueeze
from .linalg import dot, matmul, matrix_transpose, transpose
from .math import add, div, mean, mul, pow, sqrt, sub, sum, sum_to_shape
from .random import normal, uniform


def tensor(data, *, requires_grad: bool = False) -> Tensor:
    """Creates a new Tensor from the given data.

    Args:
        data (array_like): The data for the tensor.
        requires_grad (bool): If True, gradients will be computed for this tensor.
    """
    return Tensor(data, requires_grad)


__all__ = [
    "tensor",
    "Tensor",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "sum",
    "sum_to_shape",
    "mean",
    "dot",
    "matmul",
    "transpose",
    "matrix_transpose",
    "zeros",
    "ones",
    "fill",
    "contiguous",
    "reshape",
    "squeeze",
    "unsqueeze",
    "uniform",
    "normal",
]
