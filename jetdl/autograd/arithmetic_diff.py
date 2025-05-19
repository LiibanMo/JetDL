from typing import TYPE_CHECKING

from ._utils import _obtain_broadcast_booleans
from .function import Function

if TYPE_CHECKING:
    from tensorlite.tensor.tensor import Tensor


class AddBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        broadcasted, B_is_tensor = _obtain_broadcast_booleans(tensorA, tensorB)

        if broadcasted and B_is_tensor:
            if tensorA.requires_grad:
                gradA = self.result_tensor.grad.sum_to_size(tensorA.shape)

            if tensorB.requires_grad:
                gradB = self.result_tensor.grad_sum_to_size(tensorB.shape)

        elif broadcasted and not B_is_tensor:
            if tensorA.requires_grad:
                gradA = self.result_tensor.grad

        else:
            if tensorA.requires_grad:
                gradA = self.result_tensor.grad

            if tensorB.requires_grad:
                gradB = self.result_tensor.grad

        return gradA, gradB


class SubBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        broadcasted, B_is_tensor = _obtain_broadcast_booleans(tensorA, tensorB)

        if broadcasted and B_is_tensor:
            if tensorA.requires_grad:
                gradA = self.result_tensor.grad.sum_to_size(tensorA.shape)

            if tensorB.requires_grad:
                gradB = -self.result_tensor.grad.sum_to_size(tensorB.shape)

        elif broadcasted and not B_is_tensor:
            if tensorA.requires_grad:
                if isinstance(self.result_tensor.grad, Tensor):
                    gradA = self.result_tensor.grad.sum_to_size(tensorA.shape)

        else:
            if tensorA.requires_grad:
                gradA = self.result_tensor.grad

            if tensorB.requires_grad:
                gradB = -self.result_tensor.grad

        return gradA, gradB


class MulBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        broadcasted, B_is_tensor = _obtain_broadcast_booleans(tensorA, tensorB)

        if broadcasted and B_is_tensor:
            if tensorA.requires_grad:
                gradA = (tensorB * self.result_tensor.grad).sum_to_size(tensorA.shape)

            if tensorB.requires_grad:
                gradB = (tensorA * self.result_tensor.grad).sum_to_size(tensorB.shape)

        elif broadcasted and not B_is_tensor:
            if tensorA.requires_grad:
                print(f"tensorB = {tensorB}")
                gradA = tensorB * self.result_tensor.grad

        else:
            gradA = tensorB * self.result_tensor.grad if tensorA.requires_grad else None
            gradB = tensorA * self.result_tensor.grad if tensorB.requires_grad else None

        return gradA, gradB


class DivBackward(Function):
    pass


class PowBackward(Function):
    def __init__(self, tensorA, exponent, result_tensor):
        super().__init__(tensorA, exponent, result_tensor)

    def backward(self):
        tensorA, exponent = self.next_tensors
        gradA = gradB = None

        if tensorA.requires_grad:
            gradA = exponent * tensorA ** (exponent - 1)

        return gradA, gradB
