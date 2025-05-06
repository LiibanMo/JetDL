from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.01):
        super().__init__(params, lr)

    def step(self):
        from ..autograd.control_utils import no_grad

        with no_grad():
            for idx, param in enumerate(self.params):
                self.params[idx] = param - self.lr * param.grad

