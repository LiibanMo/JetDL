from tensorlite.tensor import Tensor

from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        "y = xA^T + b"
        super().__init__()
        self.weights = Parameter([[1] * out_features] * in_features)
        self.bias = Parameter([1] * out_features)

    def forward(self, input: Tensor):
        return input @ self.weights + self.bias
