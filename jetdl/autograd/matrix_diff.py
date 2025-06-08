import ctypes
from ._utils import _obtain_broadcasted_batch_dims
from .function import Function
from ..tensor._C import C_Lib, C_Tensor
from ..tensor._utils import _C_to_Python_create_tensor


class MmBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        if tensorA.ndim == 1 and tensorB.ndim == 1:
            if self.result_tensor.grad.ndim != 0:
                raise ValueError(f"Incoming gradient of the result of [{tensorA.ndim}] @ [{tensorB.ndim}] is not consistent. Got {self.result_tensor.grad.ndim}.")

            if tensorA.requires_grad:
                gradA = self.result_tensor.grad * tensorB

            if tensorB.requires_grad:
                gradB = self.result_tensor.grad * tensorA

        elif tensorA.ndim == 2 and tensorB.ndim == 1:
            if self.result_tensor.grad.ndim != 1:
                raise ValueError(f"Incoming gradient of the result of [{tensorA.ndim}] @ [{tensorB.ndim}] is not consistent. Got {self.result_tensor.grad.ndim}.")
            
            if tensorA.requires_grad:
                from ..linalg.matrix_products import outer
                gradA = outer(self.result_tensor.grad, tensorB)

            if tensorB.requires_grad:
                gradB = tensorA.T @ self.result_tensor.grad

        elif tensorA.ndim == 1 and tensorB.ndim == 2:
            if self.result_tensor.grad.ndim != 1:
                raise ValueError(f"Incoming gradient of the result of [{tensorA.ndim}] @ [{tensorB.ndim}] is not consistent. Got {self.result_tensor.grad.ndim}.")
            
            if tensorA.requires_grad:
                gradA = self.result_tensor.grad @ tensorB.T
            
            if tensorB.requires_grad:
                from ..linalg.matrix_products import outer
                gradB = outer(tensorA, self.result_tensor.grad)

        elif tensorA.ndim >= 2 and tensorB.ndim >= 2:
            if self.result_tensor.grad.ndim != max(tensorA.ndim, tensorB.ndim):
                raise ValueError(f"Incoming gradient of the result of [{tensorA.ndim}] @ [{tensorB.ndim}] is not consistent. Got {self.result_tensor.grad.ndim}.")

            broadcasted_batch_ndimsA, broadcasted_batch_ndimsB = _obtain_broadcasted_batch_dims(tensorA, tensorB)

            if tensorA.requires_grad:
                if broadcasted_batch_ndimsA:
                    gradA = (self.result_tensor.grad @ tensorB.mT).sum(broadcasted_batch_ndimsA)
                else:
                    gradA = self.result_tensor.grad @ tensorB.mT

            if tensorB.requires_grad:
                if broadcasted_batch_ndimsB:
                    gradB = (tensorA.mT @ self.result_tensor.grad).sum(broadcasted_batch_ndimsB)
                else:
                    gradB = tensorA.mT @ self.result_tensor.grad

        return gradA, gradB