from .._C import Tensor
from .._creation import zeros
from ..math import sqrt
from ..random import uniform


class Parameter(Tensor):
    def __init__(self, shape: list, init_type: str = "he", seed: int = 123):
        requires_grad = True

        if init_type == "he":
            n_in = shape[0]
            bound = sqrt(6 / n_in)
            data = uniform(-bound, bound, shape, seed=seed)
        elif init_type == "zero":
            data = zeros(shape, requires_grad=requires_grad)
        else:
            raise NotImplementedError(
                f"init type '{init_type}' not implemented for Parameter"
            )

        super().__init__(data, requires_grad)
