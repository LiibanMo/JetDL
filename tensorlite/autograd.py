class Function:
    def __init__(self, tensorA, tensorB, result_tensor):
        from .tensor import Tensor

        self.next_tensors = (tensorA, tensorB)
        if isinstance(tensorB, (int, float)) or tensorB is None:
            self.next_functions = (tensorA.grad_fn, None)
        else:
            self.next_functions = (tensorA.grad_fn, tensorB.grad_fn)
        self.result_tensor = result_tensor

    def backward(self):
        raise NotImplementedError("Backward method must be implemented.")


class AddBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        broadcasted, B_is_tensor = _obtain_broadcast_booleans(tensorA, tensorB)

        if broadcasted:
            gradA = (
                self.result_tensor.grad.sum_to_size(tensorA.shape)
                if tensorA.requires_grad
                else None
            )

            if B_is_tensor:
                if tensorB.requires_grad:
                    gradB = (
                        self.result_tensor.grad.sum_to_size(tensorB.shape)
                        if tensorB.requires_grad
                        else None
                    )

        else:
            gradA = self.result_tensor.grad if tensorA.requires_grad else None
            gradB = self.result_tensor.grad if tensorB.requires_grad else None

        return gradA, gradB


class SubBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        broadcasted, B_is_tensor = _obtain_broadcast_booleans(tensorA, tensorB)

        if broadcasted and not B_is_tensor:
            gradA = (
                self.result_tensor.grad.sum_to_size(tensorA.shape)
                if tensorA.requires_grad
                else None
            )

        elif broadcasted and B_is_tensor:
            gradA = (
                self.result_tensor.grad.sum_to_size(tensorA.shape)
                if tensorA.requires_grad
                else None
            )
            gradB = (
                -self.result_tensor.grad.sum_to_size(tensorB.shape)
                if tensorB.requires_grad
                else None
            )

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

        if broadcasted:
            gradA = (
                (tensorB * self.result_tensor.grad).sum_to_size(tensorA.shape)
                if tensorA.requires_grad
                else None
            )
            if B_is_tensor:
                gradB = (
                    (tensorA * self.result_tensor.grad).sum_to_size(tensorB.shape)
                    if tensorB.requires_grad
                    else None
                )

        else:
            gradA = tensorB * self.result_tensor.grad if tensorA.requires_grad else None
            gradB = tensorA * self.result_tensor.grad if tensorB.requires_grad else None

        return gradA, gradB


class MmBackward(Function):
    def __init__(self, tensorA, tensorB, result_tensor):
        super().__init__(tensorA, tensorB, result_tensor)

    def backward(self):
        tensorA, tensorB = self.next_tensors
        gradA = gradB = None

        if tensorA.requires_grad:
            if self.result_tensor.grad.ndim == 1 and tensorB.ndim == 1:
                from .routines import outer

                gradA = outer(self.result_tensor.grad, tensorB)

            else:
                gradA = self.result_tensor.grad @ tensorB.T

        if tensorB.requires_grad:
            if tensorA.ndim == 1 and self.result_tensor.grad.ndim == 1:
                from .routines import outer

                gradB = outer(tensorA, self.result_tensor.grad)

            else:
                gradB = tensorA.T @ self.result_tensor.grad

        return gradA, gradB


class PowBackward(Function):
    def __init__(self, tensorA, exponent, result_tensor):
        super().__init__(tensorA, exponent, result_tensor)

    def backward(self):
        tensorA, exponent = self.next_tensors
        gradA = gradB = None

        if tensorA.requires_grad:
            gradA = exponent * tensorA ** (exponent - 1)

        return gradA, gradB


class MeanBackward(Function):
    def __init__(self, tensorA, axis, result_tensor):
        super().__init__(tensorA, axis, result_tensor)

    def backward(self):
        tensorA, _ = self.next_tensors
        gradA = gradB = None

        if tensorA.requires_grad:
            gradA = tensorA / tensorA.size

        return gradA, gradB


def _obtain_broadcast_booleans(tensorA, tensorB):
    from .tensor import Tensor

    if not isinstance(tensorB, Tensor):
        broadcasted = True
        B_is_tensor = False
    elif tensorA.shape != tensorB.shape:
        broadcasted = True
        B_is_tensor = True
    else:
        broadcasted = False
        B_is_tensor = True
    return broadcasted, B_is_tensor
