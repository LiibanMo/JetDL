class Function:
    def __init__(self, tensorA, tensorB, result_tensor):
        self.next_tensors = (tensorA, tensorB)
        if isinstance(tensorB, (int, float)) or tensorB is None:
            self.next_functions = (tensorA.grad_fn, None)
        else:
            self.next_functions = (tensorA.grad_fn, tensorB.grad_fn)
        self.result_tensor = result_tensor

    def backward(self):
        raise NotImplementedError("Backward method must be implemented.")
