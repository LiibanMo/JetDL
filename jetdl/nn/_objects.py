from abc import abstractmethod

from .._C import Tensor
from .._creation import zeros
from ..math import sqrt
from ..random import uniform


class Parameter(Tensor):
    """A type of Tensor that is to be considered a module parameter.

    Parameters are Tensor subclasses that have `requires_grad` set to `True` by default.
    When used with a `Module`, they are automatically added to the list of its parameters,
    and will appear in `parameters()` iterator.
    """

    def __init__(
        self,
        shape: list,
        init_type: str = "he",
        seed: int = 123,
        device: str = "cpu",
    ) -> None:
        """Initializes a new Parameter.

        Args:
            shape (list): The shape of the parameter tensor.
            init_type (str, optional): The initialization type. Defaults to "he".
                "he": He initialization.
                "zero": Zero initialization.
            seed (int, optional): The random seed for initialization. Defaults to 123.
            device (str, optional): The device to create the parameter on. Defaults to "cpu".
        """
        requires_grad = True

        if init_type == "he":
            n_in = shape[0]
            bound = sqrt(6 / n_in)
            data = uniform(-bound, bound, shape, seed=seed, device=device)
        elif init_type == "zero":
            data = zeros(shape, requires_grad=requires_grad, device=device)
        else:
            raise NotImplementedError(
                f"init type '{init_type}' not implemented for Parameter"
            )

        super().__init__(data, requires_grad)


class Module:
    """Base class for all neural network modules.

    Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in a tree structure.
    You can assign the submodules as regular attributes.
    """

    def __init__(self) -> None:
        """Initializes the module, setting up internal dictionaries for modules and parameters."""
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_params", {})

    def __setattr__(self, name, value) -> None:
        """Sets an attribute on the module, registering submodules and parameters."""
        object.__setattr__(self, name, value)
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._params[name] = value

    def __call__(self, *inputs) -> Tensor:
        """Enables calling the module like a function, which invokes the forward pass."""
        return self.forward(*inputs)

    def parameters(self) -> iter:
        """Returns an iterator over the module's parameters.

        This includes the parameters of this module and all its submodules.
        """
        for param in self._params.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()

    def to(self, device: str) -> "Module":
        """Moves all parameters and submodules to the specified device.

        Args:
            device (str): The target device ('cpu' or 'cuda').

        Returns:
            Module: self, for method chaining.
        """
        # Move all parameters to the target device
        for name, param in self._params.items():
            new_param = param.to(device)
            object.__setattr__(self, name, new_param)
            self._params[name] = new_param

        # Recursively move all submodules
        for module in self._modules.values():
            module.to(device)

        return self

    def cuda(self, device_id: int = 0) -> "Module":
        """Moves all parameters and submodules to CUDA.

        Args:
            device_id (int, optional): The CUDA device index. Defaults to 0.

        Returns:
            Module: self, for method chaining.
        """
        device = f"cuda:{device_id}" if device_id > 0 else "cuda"
        return self.to(device)

    def cpu(self) -> "Module":
        """Moves all parameters and submodules to CPU.

        Returns:
            Module: self, for method chaining.
        """
        return self.to("cpu")

    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        """Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError("forward method not implement for nn.Module")
