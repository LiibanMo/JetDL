from typing import Union

from .._Cpp import TensorBase

Numeric = Union[list[Union[int, float]], int, float]

class Tensor(TensorBase):
    def __init__(self: "Tensor", data: "Numeric", requires_grad: bool) -> None:
        if not isinstance(data, (list, int, float)):
            raise TypeError(f"new(): invalid input type '{type(data)}'")
        super().__init__(data, requires_grad)

    def __add__(self: "Tensor", other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            operand = Tensor(other, False)
        else:
            operand = other

        from ..math import add

        return add(self, operand)
    
    def __sub__(self: "Tensor", other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            operand = Tensor(other, False)
        else:
            operand = other
        
        from ..math import sub

        return sub(self, operand)
    
    def __mul__(self: "Tensor", other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            operand = Tensor(other, False)
        else:
            operand = other
        
        from ..math import mul

        return mul(self, operand)
    
    def __truediv__(self: "Tensor", other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            operand = Tensor(other, False)
        else:
            operand = other
        
        from ..math import div

        return div(self, operand)

    def __radd__(self: "Tensor", other: Union[int, float]) -> "Tensor":
        return self + other

    def __rsub__(self: "Tensor", other: Union[int, float]) -> "Tensor":
        operand = Tensor(other, False)
        from ..math import sub
        return sub(operand, self)

    def __rmul__(self: "Tensor", other: Union[int, float]) -> "Tensor":
        return self * other

    def __rtruediv__(self: "Tensor", other: Union[int, float]) -> "Tensor":
        operand = Tensor(other, False)
        from ..math import div
        return div(operand, self)

    def __matmul__(self: "Tensor", other: "Tensor") -> "Tensor":
        from ..linalg import matmul

        return matmul(self, other)

    @property
    def T(self: "Tensor") -> "Tensor":
        from ..linalg import transpose

        return transpose(self)
    
    @property
    def mT(self: "Tensor") -> "Tensor":
        from ..linalg import matrix_transpose

        return matrix_transpose(self)


def tensor(data: "Numeric", requires_grad: bool = False) -> Tensor:
    """
    Initialize a Tensor object with the given data and gradient tracking setting.

    Args:
        data (NumericList): Input data as a nested list structure. Can contain integers or floats.
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
