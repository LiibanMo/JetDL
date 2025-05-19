from ..tensor import Tensor
from .module import Module


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        return ((input - target) ** 2).mean()
