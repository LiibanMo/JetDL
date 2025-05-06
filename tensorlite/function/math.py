import ctypes

from ..tensor import Tensor
from ..tensor._C import C_Tensor, _TensorBase
from ..tensor._utils import _C_to_Python_create_tensor


def exp(input: "Tensor") -> "Tensor":
    _TensorBase._C.c_exp.argtypes = [
        ctypes.POINTER(C_Tensor),
    ]
    _TensorBase._C.c_exp.restype = ctypes.POINTER(C_Tensor)

    c_result_tensor = _TensorBase._C.c_exp(input._tensor)

    result_tensor = _C_to_Python_create_tensor(c_result_tensor, input, None)

    return result_tensor


def log(input: "Tensor") -> "Tensor":
    for entry in input._data:
        if entry <= 0:
            raise ValueError(f"Input must have positive entries for logarithm.")

    _TensorBase._C.c_log.argtypes = [
        ctypes.POINTER(C_Tensor),
    ]
    _TensorBase._C.c_log.restype = ctypes.POINTER(C_Tensor)

    c_result_tensor = _TensorBase._C.c_log(input._tensor)

    result_tensor = _C_to_Python_create_tensor(c_result_tensor, input, None)

    return result_tensor
