class Optimizer:
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr

    def step(self):
        raise NotImplementedError(
            f"Step method of Optimizer subclass must be implemented."
        )

    def zero_grad(self, set_to_none: bool = False):
        for param in self.params:
            param.grad = None if set_to_none else 0.0


class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.01):
        super().__init__(params, lr)

    def step(self):
        for idx, param in enumerate(self.params):
            param.data -= self.lr * param.grad
            self.params[idx] = param
