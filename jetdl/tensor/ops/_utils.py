import ctypes

from .._C import C_Lib, C_Tensor
from .._utils import _C_to_Python_create_tensor


class ReductionUtils:
    @staticmethod
    def sum_axes_recursively(tensor, axes: list, idx: int):
        def sum_axes_recursively(tensor, axes: list, idx: int):
            C_Lib._C.sum_axis_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_int,
            ]
            C_Lib._C.sum_axis_tensor.restype = ctypes.POINTER(C_Tensor)
    
            if idx > 0:
                c_axis = ctypes.c_int(axes[idx])
                c_result_tensor = C_Lib._C.sum_axis_tensor(tensor._tensor, c_axis)
                result_tensor = _C_to_Python_create_tensor(c_result_tensor)
                return sum_axes_recursively(result_tensor, axes, idx - 1)
            else:
                c_axis = ctypes.c_int(axes[idx])
                c_result_tensor = C_Lib._C.sum_axis_tensor(tensor._tensor, c_axis)
                return _C_to_Python_create_tensor(c_result_tensor)

        return sum_axes_recursively(tensor, axes, idx)
