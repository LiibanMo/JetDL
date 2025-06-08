import ctypes

from ..tensor import Tensor
from ..tensor._C import C_Lib, C_Tensor
from ..tensor._utils import _C_to_Python_create_tensor


def outer(operandA: Tensor, operandB: Tensor) -> Tensor:
    C_Lib._C.outer.argtypes = [
        ctypes.POINTER(C_Tensor),
        ctypes.POINTER(C_Tensor),
    ]
    C_Lib._C.outer.restype = ctypes.POINTER(C_Tensor)

    c_result_tensor = C_Lib._C.outer(operandA._tensor, operandB._tensor)

    result_tensor = _C_to_Python_create_tensor(c_result_tensor)

    return result_tensor
