from typing import Union, TypeAlias

Numeric : TypeAlias = Union[int, float, list[int, float]] 

from .._C import (
    TensorBase, c_destroy_tensor,
)

class Tensor(TensorBase):   
    def __matmul__(self: "Tensor", other: "Tensor") -> "Tensor":
        from ..linalg import matmul
        return matmul(self, other)
    
    def __add__(self: "Tensor", other: Numeric) -> "Tensor":
        from ..linalg import add
        if isinstance(other, Tensor):
            return add(self, other)
        elif isinstance(other, Numeric):
            return add(self, Tensor(other))
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
            
    def __radd__(self: "Tensor", other: Numeric) -> "Tensor":
        return self.__add__(other)
    
    def __sub__(self: "Tensor", other: "Tensor") -> "Tensor":
        from ..linalg import sub
        if isinstance(other, Tensor):
            return sub(self, other)
        elif isinstance(other, Numeric):
            return sub(self, Tensor(other)) 
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")
    
    def __rsub__(self: "Tensor", other: Numeric) -> "Tensor":
        return self.__sub__(other)
    
    def __mul__(self: "Tensor", other: "Tensor") -> "Tensor":
        from ..linalg import mul
        if isinstance(other, Tensor):
            return mul(self, other)
        elif isinstance(other, Numeric):
            return mul(self, Tensor(other))
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

    def __truediv__(self: "Tensor", other: "Tensor") -> "Tensor":
        from ..linalg import div
        if isinstance(other, Tensor):
            return div(self, other)
        elif isinstance(other, Numeric):
            return div(self, Tensor(other))
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")
    
    @property
    def T(self: "Tensor") -> "Tensor":
        from ..linalg import transpose
        return transpose(self)
    
    @property
    def mT(self: "Tensor") -> "Tensor":
        from ..linalg import matrix_transpose
        return matrix_transpose(self)
    
    def sum(self: "Tensor", axes: Numeric) -> "Tensor":
        from ..math import sum
        return sum(self, axes) 
    
def tensor(data: Numeric) -> Tensor:
    """Creates and returns a new Tensor from the given data.
    Args:
        data (Numeric): The data to initialize the tensor with, currently supporting:
            - ints
            - floats
            - tuples
            - lists.
    Returns:
        Tensor: An initialized Tensor object.
    """
    return Tensor(data)