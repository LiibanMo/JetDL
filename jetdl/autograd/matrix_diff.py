from .function import Function


class MmBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        if tensorA.requires_grad:
            if self.result_tensor.grad.ndim <= 1 and tensorB.ndim <= 1:
                from ..linalg import outer

                gradA = outer(self.result_tensor.grad, tensorB)

            else:
                gradA = self.result_tensor.grad @ tensorB.T

        if tensorB.requires_grad:
            if tensorA.ndim <= 1 and self.result_tensor.grad.ndim <= 1:
                from ..linalg import outer

                gradB = outer(tensorA, self.result_tensor.grad)

            else:
                gradB = tensorA.T @ self.result_tensor.grad

        return gradA, gradB
