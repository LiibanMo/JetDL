import ctypes
from typing import Union

from ..tensor import Tensor
from ..tensor._C import C_Tensor, _TensorBase
from ..tensor._utils import _C_to_Python_create_tensor


def ones(shape: Union[int, list, tuple]) -> Tensor:
    if not isinstance(shape, (int, tuple, list)):
        raise TypeError(f"{type(shape)} cannot be intepreted as int, tuple or list.")

    if isinstance(shape, int):
        input_shape = [shape]
    else:
        input_shape = shape

    ndim = len(input_shape)

    c_shape = (ndim * ctypes.c_int)(*input_shape)
    c_ndim = ctypes.c_int(ndim)

    _TensorBase._C.ones.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _TensorBase._C.ones.restype = ctypes.POINTER(C_Tensor)

    c_result_tensor = _TensorBase._C.ones(c_shape, c_ndim)

    result_tensor = _C_to_Python_create_tensor(c_result_tensor, None, None)

    return result_tensor
