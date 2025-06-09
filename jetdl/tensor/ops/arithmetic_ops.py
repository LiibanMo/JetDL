import ctypes
from typing import TYPE_CHECKING, Union

from ...autograd import (AddBackward, DivBackward, MulBackward, PowBackward,
                         SubBackward)
from .._C import C_Lib, C_Tensor, _TensorBase
from .._utils import _assign_grad_and_grad_fn, _C_to_Python_create_tensor


class AddMixin:

    @staticmethod
    def tensor_add_scalar(
        operandA: _TensorBase, operandB: Union[int, float]
    ) -> _TensorBase:
        C_Lib._C.scalar_add_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_float,
        ]
        C_Lib._C.scalar_add_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.scalar_add_tensor(
            operandA._tensor, operandB
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, None, result_tensor, AddBackward)

        return result_tensor

    @staticmethod
    def tensor_add_tensor(operandA: _TensorBase, operandB: _TensorBase) -> _TensorBase:
        C_Lib._C.add_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.add_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.add_tensor(operandA._tensor, operandB._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, AddBackward)

        return result_tensor

    @staticmethod
    def tensor_add_tensor_broadcasted(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        C_Lib._C.add_broadcasted.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.add_broadcasted.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.add_broadcasted(
            operandA._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, AddBackward)

        return result_tensor


class SubMixin:

    @staticmethod
    def tensor_sub_scalar(
        operandA: _TensorBase, operandB: Union[int, float]
    ) -> _TensorBase:
        C_Lib._C.scalar_sub_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_float,
        ]
        C_Lib._C.scalar_sub_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.scalar_sub_tensor(operandA._tensor, operandB)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, None, result_tensor, SubBackward)

        return result_tensor

    @staticmethod
    def tensor_sub_tensor(operandA: _TensorBase, operandB: _TensorBase) -> _TensorBase:
        C_Lib._C.sub_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.sub_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.sub_tensor(operandA._tensor, operandB._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, SubBackward)

        return result_tensor

    @staticmethod
    def tensor_sub_tensor_broadcasted(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        C_Lib._C.sub_broadcasted.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.sub_broadcasted.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.sub_broadcasted(
            operandA._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, SubBackward)

        return result_tensor


class MulMixin:

    @staticmethod
    def tensor_mul_scalar(
        operandA: _TensorBase, operandB: Union[int, float]
    ) -> _TensorBase:
        C_Lib._C.scalar_mul_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_float,
        ]
        C_Lib._C.scalar_mul_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.scalar_mul_tensor(operandA._tensor, operandB)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MulBackward)

        return result_tensor

    @staticmethod
    def tensor_mul_tensor(operandA: _TensorBase, operandB: _TensorBase) -> _TensorBase:
        C_Lib._C.hadamard_mul_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.hadamard_mul_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.hadamard_mul_tensor(operandA._tensor, operandB._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MulBackward)

        return result_tensor

    @staticmethod
    def tensor_mul_tensor_broadcasted(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        C_Lib._C.mul_broadcasted.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.mul_broadcasted.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.mul_broadcasted(
            operandA._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MulBackward)

        return result_tensor


class DivMixin:

    @staticmethod
    def tensor_div_scalar(
        operandA: _TensorBase, operandB: Union[int, float]
    ) -> _TensorBase:
        C_Lib._C.scalar_div_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_float,
        ]
        C_Lib._C.scalar_div_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.scalar_div_tensor(operandA._tensor, operandB)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, None, result_tensor, DivBackward)

        return result_tensor

    @staticmethod
    def tensor_div_tensor(operandA: _TensorBase, operandB: _TensorBase) -> _TensorBase:
        C_Lib._C.div_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.div_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.div_tensor(operandA._tensor, operandB._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, DivBackward)

        return result_tensor

    @staticmethod
    def tensor_div_tensor_broadcasted(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        C_Lib._C.div_broadcasted.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        C_Lib._C.div_broadcasted.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.div_broadcasted(
            operandA._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, DivBackward)

        return result_tensor


class PowMixin:

    @staticmethod
    def tensor_pow_scalar(
        operandA: _TensorBase, operandB: Union[int, float]
    ) -> _TensorBase:
        C_Lib._C.pow_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_float,
        ]
        C_Lib._C.pow_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.pow_tensor(operandA._tensor, operandB)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(operandA, None, result_tensor, PowBackward)

        return result_tensor
