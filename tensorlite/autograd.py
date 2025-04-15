from abc import ABC, abstractmethod


class Function(ABC):
    @abstractmethod
    def backward(*tensors):
        return NotImplementedError


class AddBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        self.prev_tensors = (tensorA, tensorB)
        self.next_functions = (tensorA.grad_fn, tensorB.grad_fn)
        self.result_tensor = result_tensor

    def backward(self):
        tensorA, tensorB = self.prev_tensors
        gradA = gradB = None
        if tensorA.requires_grad:
            gradA = self.result_tensor.grad
        if tensorB.requires_grad:
            gradB = self.result_tensor.grad
        return gradA, gradB


class MmBackward(Function):
    def __init__(self, tensorA, tensorB, output_tensor):
        self.prev_tensors = (tensorA, tensorB)
        self.next_functions = (tensorA.grad_fn, tensorB.grad_fn)
        self.result_tensor = output_tensor

    def backward(self):
        tensorA, tensorB = self.prev_tensors
        if tensorA.requires_grad:
            if self.result_tensor.grad.ndim == 1 and tensorB.ndim == 1:
                gradA = None  # outer(output_grad, tensorB)
            else:
                gradA = self.result_tensor.grad @ tensorB.T
        if tensorB.requires_grad:
            if tensorA.ndim == 1 and self.result_tensor.grad.ndim == 1:
                gradB = None  # outer(output_grad, tensorB)
            else:
                gradB = tensorA.T @ self.result_tensor.grad
        return gradA, gradB
