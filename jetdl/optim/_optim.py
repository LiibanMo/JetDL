from abc import abstractmethod

from .._C.optim import c_zero_grad
from .._creation import zeros
from ..nn._layers import Module


class Optimizer:
    def __init__(self, params: iter) -> None:
        self.params = [param for param in params]

    def zero_grad(self) -> None:
        c_zero_grad(self.params)

    @abstractmethod
    def step(self):
        raise NotImplementedError("Optimizer.step not implemented.")
