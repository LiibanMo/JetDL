from abc import abstractmethod

from .._C import Tensor
from .._C.nn import c_linear_forward, c_relu_forward
from ._param import Parameter


class Module:
    def __init__(self):
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_params", {})

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._params[name] = value

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def parameters(self):
        for param in self._params.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError("forward method not implement for nn.Module")


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_type: str = "he",
        seed: int = 123,
    ):
        super().__init__()

        self.weight = Parameter([out_features, in_features], init_type, seed)
        self.bias = Parameter([out_features], "zero", seed) 

    def forward(self, input: Tensor) -> Tensor:
        return c_linear_forward(input, self.weight, self.bias)

class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        return c_relu_forward(input)