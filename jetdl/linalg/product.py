from .._C.utils import c_utils_check_dot_shapes
from .._C.linalg import c_dot
from ..tensor import Tensor

def dot(a: Tensor, b: Tensor) -> Tensor:
    c_utils_check_dot_shapes(a.shape, b.shape)
    return c_dot(a, b)