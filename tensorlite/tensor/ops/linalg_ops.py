import ctypes
from typing import TYPE_CHECKING, Union

from ...autograd import MmBackward
from .._C import C_Tensor, _TensorBase
from .._utils import _assign_grad_and_grad_fn, _C_to_Python_create_tensor


class MatmulMixin:

    @staticmethod
    def vector_matmul_vector(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        _TensorBase._C.vector_dot_product.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.vector_dot_product.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.vector_dot_product(
            operandA._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MmBackward)

        return result_tensor

    @staticmethod
    def vector_matmul_matrix(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        new_shape = [1] + operandA._shape.copy()

        broadcasted_tensor = operandA.reshape(new_shape)

        _TensorBase._C.matmul_2d_2d.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.matmul_2d_2d(
            broadcasted_tensor._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

        result_shape = result_tensor._shape.copy()
        result_shape.pop(0)

        result_tensor = result_tensor.reshape(result_shape)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MmBackward)

        return result_tensor

    @staticmethod
    def matrix_matmul_vector(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        new_shape = operandB._shape.copy() + [1]

        broadcasted_tensor = operandB.reshape(new_shape)

        _TensorBase._C.matmul_2d_2d.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.matmul_2d_2d(
            operandA._tensor, broadcasted_tensor._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

        result_shape = result_tensor._shape.copy()

        result_shape.pop(1)

        result_tensor = result_tensor.reshape(result_shape)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MmBackward)

        return result_tensor

    @staticmethod
    def matrix_matmul_matrix(
        operandA: _TensorBase, operandB: _TensorBase
    ) -> _TensorBase:
        _TensorBase._C.matmul_2d_2d.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.matmul_2d_2d(
            operandA._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MmBackward)

        return result_tensor
    
    @staticmethod
    def vector_matmul_tensor_broadcasted(operandA: _TensorBase, operandB: _TensorBase) -> _TensorBase:
        _TensorBase._C.matmul_broadcasted.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.matmul_broadcasted.restype = ctypes.POINTER(C_Tensor)
        
        new_shape = operandB._shape.copy() + [1]
        broadcasted_tensor = operandB.reshape(new_shape)

        c_result_tensor = _TensorBase._C.matmul_broadcasted(
            operandA._tensor, broadcasted_tensor._tensor
        )
        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

        result_shape = result_tensor._shape.copy()
        idx_to_squeeze = result_tensor.ndim - 1
        result_shape.pop(idx_to_squeeze)

        result_tensor = result_tensor.reshape(result_shape)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MmBackward)

        return result_tensor
    
    @staticmethod
    def tensor_matmul_vector_broadcasted(operandA: _TensorBase, operandB: _TensorBase) -> _TensorBase:
        _TensorBase._C.matmul_broadcasted.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.matmul_broadcasted.restype = ctypes.POINTER(C_Tensor)

        new_shape = operandB._shape.copy() + [1]
        broadcasted_tensor = operandB.reshape(new_shape)

        c_result_tensor = _TensorBase._C.matmul_broadcasted(
            operandA._tensor, broadcasted_tensor._tensor
        )
        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

        result_shape = result_tensor._shape.copy()
        idx_to_squeeze = result_tensor.ndim - 1
        result_shape.pop(idx_to_squeeze)

        result_tensor = result_tensor.reshape(result_shape)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MmBackward)

        return result_tensor
    
    @staticmethod
    def tensor_matmul_tensor_broadcasted(operandA: _TensorBase, operandB: _TensorBase) -> _TensorBase:
        _TensorBase._C.matmul_broadcasted.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.matmul_broadcasted.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.matmul_broadcasted(
            operandA._tensor, operandB._tensor
        )

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, operandB)

        _assign_grad_and_grad_fn(operandA, operandB, result_tensor, MmBackward)

        return result_tensor


class TransposeMixin:
    @staticmethod
    def T(operandA: _TensorBase) -> _TensorBase:
        _TensorBase._C.transpose_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.transpose_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.transpose_tensor(operandA._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        result_tensor._data = result_tensor.flatten()._data

        return result_tensor

    @staticmethod
    def mT(operandA: _TensorBase) -> _TensorBase:
        _TensorBase._C.matrix_transpose.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        _TensorBase._C.matrix_transpose.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = _TensorBase._C.matrix_transpose(operandA._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor, operandA, None)

        result_tensor._data = result_tensor.flatten()._data

        return result_tensor
