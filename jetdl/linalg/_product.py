from .._C.linalg import c_dot, c_matmul
from .._tensor import Tensor

def dot(a: Tensor, b: Tensor) -> Tensor:
    return c_dot(a, b)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return c_matmul(a, b)