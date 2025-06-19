import ctypes
from typing import Union, Optional

from ._C import _TensorBase
from ._utils import (_build_comp_graph, _C_to_Python_create_tensor,
                     _can_broadcast)
from .ops import *


class Tensor(_TensorBase):
    def __init__(self, data:Union[None, int, float, list]=None, requires_grad=True):
        super().__init__(data, requires_grad)

    def __add__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            return AddMixin.tensor_add_scalar(self, other)
        elif isinstance(other, Tensor):
            if self._shape == other._shape:
                return AddMixin.tensor_add_tensor(self, other)
            elif _can_broadcast(self._shape, self.ndim, other._shape, other.ndim):
                return AddMixin.tensor_add_tensor_broadcasted(self, other)
            else:
                raise RuntimeError(
                    f"Cannot add tensors with shapes {self.shape} and {other.shape}."
                )
    
    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)
    
    def __sub__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            return SubMixin.tensor_sub_scalar(self, other)
        elif isinstance(other, Tensor):
            if self._shape == other._shape:
                return SubMixin.tensor_sub_tensor(self, other)
            elif _can_broadcast(self._shape, self.ndim, other._shape, other.ndim):
                return SubMixin.tensor_sub_tensor_broadcasted(self, other)
            else:
                raise RuntimeError(
                    f"Cannot subtract tensors with shapes {self.shape} and {other.shape}."
                )
            
    def __rsub__(self, other) -> "Tensor":
        return (self.__mul__(-1)).__add__(other) 

    def __mul__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            return MulMixin.tensor_mul_scalar(self, other)
        elif isinstance(other, Tensor):
            if self._shape == other._shape:
                return MulMixin.tensor_mul_tensor(self, other)
            elif _can_broadcast(self._shape, self.ndim, other._shape, other.ndim):
                return MulMixin.tensor_mul_tensor_broadcasted(self, other)
            else:
                raise RuntimeError(
                    f"Cannot multiply tensors with shapes {self.shape} and {other.shape}."
                )
    
    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)
    
    def __neg__(self) -> "Tensor":
        return self.__mul__(-1)
    
    def __truediv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            return DivMixin.tensor_div_scalar(self, other)
        elif isinstance(other, Tensor):
            if self._shape == other._shape:
                return DivMixin.tensor_div_tensor(self, other)
            elif _can_broadcast(self._shape, self.ndim, other._shape, other.ndim):
                return DivMixin.tensor_div_tensor_broadcasted(self, other)
            else:
                raise RuntimeError(
                    f"Cannot divide tensors with shapes {self.shape} and {other.shape}."
                )
    
    def __rtruediv__(self, other) -> "Tensor":
        return (self.__pow__(-1)).__mul__(other)
        
    def __matmul__(self, other: "Tensor") -> "Tensor":
        runtime_error = RuntimeError(
                            f"Cannot perform matrix multiplication on tensors with shapes {self.shape} and {other.shape}."
                        )
        if self.ndim == 1 and other.ndim == 1:
            if self._shape == other._shape:
                return MatmulMixin.vector_matmul_vector(self, other)
            else:
                raise runtime_error
            
        elif self.ndim == 1 and other.ndim == 2:
            if self._shape[0] == other._shape[0]:
                return MatmulMixin.vector_matmul_matrix(self, other)
            else:
                raise runtime_error
        
        elif self.ndim == 2 and other.ndim == 1:
            if self._shape[1] == other._shape[0]:
                return MatmulMixin.matrix_matmul_vector(self, other)
            else:
                raise runtime_error
            
        elif self.ndim == 2 and other.ndim == 2:
            if self._shape[1] == other._shape[0]:
                return MatmulMixin.matrix_matmul_matrix(self, other)
            else:
                raise runtime_error
            
        elif self.ndim > 2 or other.ndim > 2:
            if self.ndim == 1:
                if self._shape[0] == other._shape[-2]:
                    return MatmulMixin.vector_matmul_tensor_broadcasted(self, other)
                else:  
                    raise runtime_error
                
            elif other.ndim == 1:
                if self._shape[-1] == other._shape[0]:
                    return MatmulMixin.tensor_matmul_vector_broadcasted(self, other)
                else:
                    raise runtime_error
            
            else:
                if self._shape[-1] == other._shape[-2]:
                    return MatmulMixin.tensor_matmul_tensor_broadcasted(self, other)
                else:
                    raise runtime_error
        else:
            raise ValueError(
                f"Invalid dimensions for matmul. Got {self.ndim} and {other.ndim}."
            )
        
    def __pow__(self, other: Union[int, float]) -> "Tensor":
        return PowMixin.tensor_pow_scalar(self, other)

    @property
    def shape(self) -> tuple:
        return tuple(self._shape) if self.ndim > 0 else ()

    def flatten(self) -> "Tensor":
        return ShapeMixin.flatten(self)

    def reshape(self, shape: Union[list, tuple]) -> "Tensor":
        if isinstance(shape, tuple):
            input_shape = list(shape)
        else:
            input_shape = shape
        return ShapeMixin.reshape(self, input_shape)

    @property
    def T(self) -> "Tensor":
        if self.ndim < 2:
            return self
        return TransposeMixin.T(self)

    @property
    def mT(self) -> "Tensor":
        if self.ndim < 2:
            return self
        return TransposeMixin.mT(self)

    def sum(self, axis: Union[None, int, list, tuple] = None) -> "Tensor":
        if axis is None:
            return SumMixin.total_sum(self)
        
        elif isinstance(axis, int):
            if axis not in range(self.ndim):
                raise IndexError(f"The inputted axis {axis} is out of range.")
            return SumMixin.sum_axis(self, axis)
        
        elif isinstance(axis, (list, tuple)):
            input_axes = list(axis)
            for axis in input_axes:
                if axis not in range(self.ndim):
                    raise IndexError(f"The inputted axis {axis} is out of range.")
            
            return SumMixin.sum_axes(self, input_axes)

    def mean(self, axis: Union[None, int, list, tuple] = None) -> "Tensor":
        if axis is None:
            return MeanMixin.total_mean(self)
        elif isinstance(axis, int):
            return MeanMixin.mean_axis(self, axis)

    def copy(self) -> "Tensor":
        c_data = (len(self._data) * ctypes.c_float)(*self._data)
        c_shape = (ctypes.c_int * len(self._shape))(*self._shape)
        c_ndim = ctypes.c_int(len(self._shape))

        c_result_tensor = self._C.create_tensor(
            c_data,
            c_shape,
            c_ndim,
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        return result_tensor

    def sum_to_size(self, size: Union[int, list, tuple]) -> "Tensor":
        if _can_broadcast:
            return SumMixin.sum_to_size(self, size)
        else:
            raise RuntimeError(
                f"size {tuple(self._shape)} cannot reduce to {tuple(size)}."
            )

    def backward(self):
        from ..function import ones

        self.grad = ones(self.shape)

        topo = _build_comp_graph(self)

        for fn in reversed(topo):
            gradients = fn.backward()
            next_tensors = fn.next_tensors
            for grad, tensor in zip(gradients, next_tensors):
                if grad and isinstance(tensor, Tensor):
                    tensor.grad += grad

def tensor(data:Optional[list], requires_grad:bool=True) -> Tensor:
    """
    Create a tensor from the given data.

    Args:
        data (Optional[list]): The data to create the tensor from.
        requires_grad (bool): Whether to track gradients for this tensor.

    Returns:
        Tensor: The created tensor.
    """
    return Tensor(data, requires_grad)