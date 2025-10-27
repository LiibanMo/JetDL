from abc import abstractmethod

from .._C.optim import c_zero_grad


class Optimizer:
    """Base class for all optimizers.

    Args:
        params (iterable): an iterable of `Parameter` s or `dict` s.
    """

    def __init__(self, params: iter) -> None:
        """Initializes the Optimizer.

        Args:
            params (iter): An iterable of parameters to optimize.
        """
        self.params = [param for param in params]

    def zero_grad(self) -> None:
        """Sets the gradients of all optimized tensors to null."""
        c_zero_grad(self.params)

    @abstractmethod
    def step(self):
        """Performs a single optimization step.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Optimizer.step not implemented.")
