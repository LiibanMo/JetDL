from .._C.optim import c_sgd
from ._optim import Optimizer


class SGD(Optimizer):
    def __init__(self, params: iter, lr: float = 1e-3):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            c_sgd(param, self.lr)
