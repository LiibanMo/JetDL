import pytest

from tensorlite.nn import Linear, Module, MSELoss, Parameter
from tensorlite.tensor import Tensor


def test_parameter_initialization():
    tensor = Tensor([1, 2, 3])
    param = Parameter(tensor)
    assert param.data == tensor
    assert param.requires_grad is True
    assert param.grad == 0.0


def test_parameter_setattr():
    param = Parameter([1, 2, 3])
    param.grad = 5.0
    assert param.grad == 5.0


def test_module_initialization():
    module = Module()
    assert len(module._modules) == 0
    assert len(module._parameters) == 0


def test_module_setattr():
    module = Module()
    param = Parameter([1, 2, 3])
    module.param = param
    assert "param" in module._parameters
    assert module._parameters["param"] == param


def test_linear_forward():
    linear = Linear(2, 3)
    input_tensor = Tensor([[1, 2]])
    output = linear(input_tensor)
    assert output.shape == (1, 3)


def test_mse_loss_forward():
    mse_loss = MSELoss()
    input_tensor = Tensor([1, 2, 3])
    target_tensor = Tensor([1, 2, 3])
    loss = mse_loss(input_tensor, target_tensor)
    assert loss.data[0] == 0.0
    assert loss.shape == ()
    assert loss.ndim == 0


def test_module_parameters():
    linear = Linear(2, 3)
    params = list(linear.parameters())
    assert len(params) == 2
    assert isinstance(params[0], Parameter)
    assert isinstance(params[1], Parameter)


def test_module_zero_grad():
    linear = Linear(2, 3)
    for param in linear.parameters():
        param.grad = 1.0
    linear.zero_grad()
    for param in linear.parameters():
        assert param.grad == 0.0
