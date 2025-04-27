import ctypes
import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .tensor import Tensor


class _C_Routines:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "libtensor.so")
    _C = ctypes.CDLL(lib_path)


def ones(shape) -> "Tensor":
    from .tensor import C_Tensor, _C_to_Python_create_tensor

    if not isinstance(shape, (int, tuple, list)):
        raise TypeError(f"{type(shape)} cannot be intepreted as int, tuple or list.")

    if isinstance(shape, int):
        shape = [shape]

    ndim = len(shape)

    c_shape = (ndim * ctypes.c_int)(*shape)
    c_ndim = ctypes.c_int(ndim)

    _C_Routines._C.ones.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _C_Routines._C.ones.restype = ctypes.POINTER(C_Tensor)

    c_result_tensor = _C_Routines._C.ones(c_shape, c_ndim)

    return _C_to_Python_create_tensor(c_result_tensor)


def outer(tensorA: "Tensor", tensorB: "Tensor") -> "Tensor":
    from .tensor import C_Tensor, _C_to_Python_create_tensor

    _C_Routines._C.outer.argtypes = [
        ctypes.POINTER(C_Tensor),
        ctypes.POINTER(C_Tensor),
    ]
    _C_Routines._C.outer.restype = ctypes.POINTER(C_Tensor)

    c_result_tensor = _C_Routines._C.outer(tensorA._tensor, tensorB._tensor)

    return _C_to_Python_create_tensor(c_result_tensor)


class GradMode:
    _enabled = True
    _no_grad_depth = 0

    @classmethod
    def is_enabled(cls):
        return cls.is_enabled and cls._no_grad_depth == 0

class no_grad:
    def __enter__(self):
        GradMode._no_grad_depth += 1

    def __exit__(self, exec_type, exec_value, traceback):
        GradMode._no_grad_depth -= 1