from typing import Union
from .._Cpp import c_ones

def ones(shape: Union[int, list[int], tuple[int]], requires_grad:bool = False):
    if isinstance(shape, int):
        return c_ones([shape], requires_grad)
    else:
        return c_ones(list(shape), requires_grad)