from .._C import Tensor
from ._layers import Module


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: Tensor, y_pred: Tensor):
        squared_diff = (y_true - y_pred) ** 2
        return squared_diff.mean()
