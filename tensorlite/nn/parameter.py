from typing import Union
from ..tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data: Union[None, int, float, list] = None, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)
        self.grad = 0.0

    def __str__(self):
        result_str = super().__str__()
        result_str = "Parameter(" + result_str[7:]
        return result_str

    def __repr__(self):
        return self.__str__()
