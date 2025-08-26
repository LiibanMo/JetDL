from typing import Union, TypeAlias

ArrayLike : TypeAlias = Union[int, float, list[int, float]] 

from .._C import (
    TensorBase, c_destroy_tensor,
)

class Tensor(TensorBase):   
    def __matmul__(self: "Tensor", other: "Tensor") -> "Tensor":
        from ..linalg import matmul
        return matmul(self, other)
    
def tensor(data: ArrayLike) -> Tensor:
    """Creates and returns a new Tensor from the given data.
    Args:
        data (ArrayLike): The data to initialize the tensor with, such as a list or numpy array.
    Returns:
        Tensor: An initialized Tensor object.
    """
    return Tensor(data)

tensor__all__ = [
    'Tensor', 
    'tensor',
]