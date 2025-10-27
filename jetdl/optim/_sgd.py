from .._C.optim import c_sgd
from ._optim import Optimizer


class SGD(Optimizer):
    """Implements stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
    """

    def __init__(self, params: iter, lr: float = 1e-3):
        """Initializes the SGD optimizer.

        Args:
            params (iter): An iterable of parameters to optimize.
            lr (float, optional): The learning rate. Defaults to 1e-3.
        """
        super().__init__(params)
        self.lr = lr

    def step(self):
        """Performs a single optimization step."""
        for param in self.params:
            c_sgd(param, self.lr)
