from typing import Union
from .._Cpp import (
    Tensor as TensorBase
)

numeric = Union[int, float]

class Tensor(TensorBase):
    def __init__(self, data: list[numeric], requires_grad: bool) -> None:
        super().__init__(data, requires_grad)
    
def tensor(data: list[numeric], requires_grad: bool = True) -> Tensor:
    """
        Initialize a Tensor object with the given data and gradient tracking setting.

        Args:
            data (list[numeric]): Input data as a nested list structure. Can contain integers or floats.
                                 The data will be flattened and stored internally.
            requires_grad (bool): Whether to track gradients for this tensor during backpropagation.
                                Defaults to False.

        Example:
            >>> tensor_data = [[1, 2, 3], [4, 5, 6]]
            >>> t = Tensor(tensor_data, requires_grad=True)
            >>> print(t.shape)
            (2, 3)
            >>> print(t.ndim)
            2

        Note:
            The input data is automatically flattened and stored as a contiguous array.
            Shape and strides are computed automatically based on the input structure.
    """
    return Tensor(data, requires_grad)