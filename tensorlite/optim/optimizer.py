from typing import Union
from collections import OrderedDict

from ..nn import Parameter

class Optimizer:
    def __init__(self, params:Union[OrderedDict[Parameter], Parameter], lr: float):
        if isinstance(params, OrderedDict):
            self.params = list(params)
        elif isinstance(params, Parameter):
            self.params = [params]
        else:
            raise TypeError(f"Incorrect data type passed into `params`. Expected OrderedDict[Parameter] or Parameter. Got {type(params)}.")
        
        self.lr = lr

    def step(self):
        raise NotImplementedError(
            f"Step method of Optimizer subclass must be implemented."
        )

    def zero_grad(self, set_to_none: bool = False):
        for idx in range(len(self.params)):
            self.params[idx].grad = None if set_to_none else 0.0
            print(self.params[idx].grad)
            print(self.params[idx])
            
