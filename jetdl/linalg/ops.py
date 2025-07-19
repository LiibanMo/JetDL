from .._Cpp import c_add
from ..tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    return c_add(a, b)
