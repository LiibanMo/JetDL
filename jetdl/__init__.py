from . import linalg, nn, optim
from ._C import Tensor, cuda_device_count, cuda_is_available
from ._creation import fill, ones, zeros


from ._manip import contiguous, reshape, squeeze, unsqueeze
from .linalg import dot, matmul, matrix_transpose, transpose
from .math import (abs, add, clamp, cos, cosh, div, exp, log, log2, log10,
                   mean, mul, pow, sign, sin, sinh, sqrt, sub, sum,
                   sum_to_shape, tanh)
from .random import normal, rand, uniform


def tensor(data, *, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    """Creates a new Tensor from the given data.

    Args:
        data (array_like): The data for the tensor.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        device (str): The device to create the tensor on ('cpu' or 'cuda:N').
    """
    return Tensor(data, requires_grad, device)


class cuda:
    """CUDA utilities namespace."""

    @staticmethod
    def is_available() -> bool:
        """Returns True if CUDA is available."""
        return cuda_is_available()

    @staticmethod
    def device_count() -> int:
        """Returns the number of available CUDA devices."""
        return cuda_device_count()


__all__ = [
    "cuda",
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
    "exp",
    "log",
    "log10",
    "log2",
    "sin",
    "cos",
    "tanh",
    "sinh",
    "cosh",
    "abs",
    "sign",
    "clamp",
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
    "rand",
]
