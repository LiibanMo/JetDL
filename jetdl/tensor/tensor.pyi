from typing import Union

from .._C import C_Tensor, c_init_tensor, c_destroy_tensor

class Tensor(C_Tensor):
    _data:list[int]
    ndim:int
    shape:tuple[int]
    size:int
    strides:tuple[int]
    
    def __new__(cls, data): ...
    
    def __del__(self): ...

    def __matmul__(self: "Tensor", other: "Tensor") -> "Tensor": ...