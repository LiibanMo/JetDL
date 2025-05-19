from .function import Function


class MeanBackward(Function):
    def __init__(self, tensorA, axis, result_tensor):
        super().__init__(tensorA, axis, result_tensor)

    def backward(self):
        tensorA, _ = self.next_tensors
        gradA = gradB = None

        if tensorA.requires_grad:
            gradA = tensorA / tensorA.size

        return gradA, gradB
