import ctypes
from typing import TYPE_CHECKING, Union

from .._C import C_Tensor, _TensorBase
from .._utils import _C_to_Python_create_tensor


class ShapeMixin:

    @staticmethod
    def reshape(operandA: _TensorBase, new_shape: list) -> _TensorBase:
        if new_shape == operandA._shape:
            return operandA

        _TensorBase._C.reshape_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        _TensorBase._C.reshape_tensor.restype = ctypes.POINTER(C_Tensor)

        _c_shape = (ctypes.c_int * len(new_shape))(*new_shape)

        c_result_tensor = _TensorBase._C.reshape_tensor(
            operandA._tensor, _c_shape, len(new_shape)
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        return result_tensor

    @staticmethod
    def flatten(operandA: _TensorBase) -> _TensorBase:
        _TensorBase._C.flatten_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]

        _TensorBase._C.flatten_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.flatten_tensor(operandA._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        return result_tensor
