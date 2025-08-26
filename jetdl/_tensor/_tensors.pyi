from typing import Union, TypeAlias

from .._C import TensorBase

ArrayLike : TypeAlias = Union[int, float, list[int, float]] 

class Tensor(TensorBase):
    _data:list[int]
    ndim:int
    shape:tuple[int]
    size:int
    strides:tuple[int]
    
    def __matmul__(self: "Tensor", other: "Tensor") -> "Tensor": ...
    @property
    def T(self: "Tensor") -> "Tensor": ...
    @property
    def mT(self: "Tensor") -> "Tensor": ...

def tensor(data: ArrayLike) -> Tensor: ...