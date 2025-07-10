from typing import TYPE_CHECKING
from ..tensor import Tensor
from .._Cpp import (
    c_matmul
)

def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Input(s):
        a (Tensor) : Must be 1-D i.e. a.ndim == 1
        b (Tensor) : Must be 1-D i.e. b.ndim == 1
    Output(s):
        Tensor : a scalar Tensor object
    """
    if a.shape != b.shape:
        raise ValueError(f"shapes {a.shape} and {b.shape} are not compatible")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"dot product requires 1-D tensors, got tensors with {a.ndim} and {b.ndim} dimensions")
    
    return c_matmul(a, b)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return c_matmul(a, b)