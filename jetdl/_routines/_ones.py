from typing import Union
from .._C.routines import c_ones
from .._tensor import Tensor

def ones(shape: Union[list, tuple]) -> Tensor:
    return c_ones(shape)