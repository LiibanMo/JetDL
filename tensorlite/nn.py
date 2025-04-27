from collections import OrderedDict

from .tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, requires_grad: bool = True):
        if not isinstance(data, Tensor):
            data = Tensor(data, requires_grad=requires_grad)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = 0.0

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __setattr__(self, name, value):
        if name in {"data", "requires_grad", "grad"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.data, name, value)

    def __str__(self):
        result = super().__str__()
        result = "Parameter" + result[6:]
        return result

    def __repr__(self):
        return self.__str__()


class Module:
    def __init__(self):
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_attribute_order", [])

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, Module):
            self._modules[name] = value
            item = (name, "module")
            self._attribute_order.append(item)
        elif isinstance(value, Parameter):
            self._parameters[name] = value
            item = (name, "parameter")
            self._attribute_order.append(item)

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        raise NotImplementedError("Forward method must be implemented.")

    def parameters(self):
        for name, type in self._attribute_order:
            if type == "parameter":
                yield self._parameters[name]
            if type == "module":
                yield from self._modules[name].parameters()

    def zero_grad(self, set_to_none: bool = False):
        for param in self.parameters():
            param.grad = None if set_to_none else 0.0


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        "y = xA^T + b"
        super().__init__()
        self.weights = Parameter([[1] * out_features] * in_features)
        self.bias = Parameter([1] * out_features)

    def forward(self, input: Tensor):
        return input @ self.weights + self.bias


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        return ((input - target) ** 2).mean()
