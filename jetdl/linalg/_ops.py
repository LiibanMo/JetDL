from .._C.math import c_ops
from .._tensor import Tensor

def add(a: Tensor, b: Tensor) -> Tensor:
    return c_ops(a, b, "ADD")

def sub(a: Tensor, b: Tensor) -> Tensor:
    return c_ops(a, b, "SUB");

def mul(a: Tensor, b: Tensor) -> Tensor:
    return c_ops(a, b, "MUL")

def div(a: Tensor, b: Tensor) -> Tensor:
    return c_ops(a, b, "DIV")