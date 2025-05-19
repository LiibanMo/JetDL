import ctypes
from typing import TYPE_CHECKING, Union

from ...autograd import MeanBackward
from .._C import C_Tensor, _TensorBase
from .._utils import _assign_grad_and_grad_fn, _C_to_Python_create_tensor
from ._utils import ReductionUtils


class SumMixin:

    @staticmethod
    def total_sum(operandA: _TensorBase) -> _TensorBase:
        _TensorBase._C.sum_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.sum_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.sum_tensor(operandA._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        return result_tensor

    @staticmethod
    def sum_axis(operandA: _TensorBase, axis: int) -> _TensorBase:
        if axis < 0:
            c_axis = ctypes.c_int(operandA.ndim + axis)
        else:
            c_axis = ctypes.c_int(axis)

        _TensorBase._C.sum_axis_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_int,
        ]
        _TensorBase._C.sum_axis_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.sum_axis_tensor(operandA._tensor, c_axis)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        return result_tensor

    @staticmethod
    def sum_axes(operandA: _TensorBase, input_axes: list) -> _TensorBase:
        axes = []
        for dim in input_axes:
            if dim < 0:
                axes.append(operandA.ndim + dim)
            else:
                axes.append(dim)
        axes.sort()

        _TensorBase._C.sum_axis_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_int,
        ]
        _TensorBase._C.sum_axis_tensor.restype = ctypes.POINTER(C_Tensor)

        result_tensor = ReductionUtils.sum_axes_recursively(
            operandA, axes, len(axes) - 1
        )

        return result_tensor

    @staticmethod
    def sum_to_size(
        operandA: _TensorBase, size: Union[int, list, tuple]
    ) -> _TensorBase:
        if isinstance(size, int):
            shape = [size]

        elif isinstance(size, tuple):
            shape = list(size)

        else:
            shape = size.copy()

        broadcasted_axes = []
        for idx in range(operandA.ndim):
            idx_for_new_shape = idx - operandA.ndim + len(shape)
            if idx_for_new_shape < 0:
                broadcasted_axes.append(idx)
            elif (
                idx_for_new_shape >= 0
                and shape[idx_for_new_shape] != operandA._shape[idx]
                and shape[idx_for_new_shape] == 1
            ):
                broadcasted_axes.append(idx)

        if broadcasted_axes:
            result_tensor = ReductionUtils.sum_axes_recursively(
                operandA, broadcasted_axes, len(broadcasted_axes) - 1
            )
            result_tensor = result_tensor.reshape(shape)

        return result_tensor


class MeanMixin:
    
    @staticmethod
    def total_mean(operandA: _TensorBase) -> _TensorBase:
        _TensorBase._C.mean_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.mean_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.mean_tensor(operandA._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        _assign_grad_and_grad_fn(operandA, None, result_tensor, MeanBackward)

        return result_tensor

    @staticmethod
    def mean_axis(operandA: _TensorBase, axis: int) -> _TensorBase:
        if axis < 0:
            c_axis = ctypes.c_int(operandA.ndim + axis)
        else:
            c_axis = ctypes.c_int(axis)

        _TensorBase._C.mean_axis_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_int,
        ]
        _TensorBase._C.mean_axis_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.mean_axis_tensor(operandA._tensor, c_axis)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        return result_tensor
