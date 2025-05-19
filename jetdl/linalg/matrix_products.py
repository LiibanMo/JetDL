import ctypes

from ..tensor import Tensor
from ..tensor._C import C_Tensor, _TensorBase
from ..tensor._utils import _C_to_Python_create_tensor


def outer(operandA: Tensor, operandB: Tensor) -> Tensor:
    _TensorBase._C.outer.argtypes = [
        ctypes.POINTER(C_Tensor),
        ctypes.POINTER(C_Tensor),
    ]
    _TensorBase._C.outer.restype = ctypes.POINTER(C_Tensor)

    c_result_tensor = _TensorBase._C.outer(operandA._tensor, operandB._tensor)

    result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

    return result_tensor
