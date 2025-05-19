import ctypes

from .._C import _TensorBase
from .._utils import _C_to_Python_create_tensor


class ReductionUtils:
    @staticmethod
    def sum_axes_recursively(tensor, axes: list, idx: int):
        def sum_axes_recursively(tensor, axes: list, idx: int):
            if idx > 0:
                c_axis = ctypes.c_int(axes[idx])
                c_result_tensor = _TensorBase._C.sum_axis_tensor(tensor._tensor, c_axis)
                result_tensor = _C_to_Python_create_tensor(c_result_tensor)
                return sum_axes_recursively(result_tensor, axes, idx - 1)
            else:
                c_axis = ctypes.c_int(axes[idx])
                c_result_tensor = _TensorBase._C.sum_axis_tensor(tensor._tensor, c_axis)
                return _C_to_Python_create_tensor(c_result_tensor)

        return sum_axes_recursively(tensor, axes, idx)
