from collections import OrderedDict

from .parameter import Parameter


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
