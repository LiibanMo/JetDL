import pytest

from jetdl.nn import Linear, Module, MSELoss, Parameter
from jetdl import tensor, Tensor


def test_parameter_initialization():
    result_tensor = tensor([1, 2, 3])
    param = Parameter([1, 2, 3])
    assert param._data == result_tensor._data
    assert param.requires_grad is True
    assert param.grad == pytest.approx(0.0)


def test_module_initialization():
    module = Module()
    assert hasattr(module, "_modules")
    assert hasattr(module, "_parameters")
    assert hasattr(module, "_attribute_order")


def test_linear_forward():
    linear = Linear(2, 3)
    input_tensor = tensor([[1, 2]])
    output = linear(input_tensor)
    assert output.shape == (1, 3)


def test_mse_loss_forward():
    mse_loss = MSELoss()
    input_tensor = tensor([1, 2, 3])
    target_tensor = tensor([1, 2, 4])
    loss = mse_loss(input_tensor, target_tensor)
    assert loss._data[0] == pytest.approx(1 / 3)
    assert len(loss._data) == 1


def test_module_parameters():
    linear = Linear(2, 3)
    params = list(linear.parameters())
    assert len(params) == 2
    assert isinstance(params[0], Parameter)
    assert isinstance(params[1], Parameter)


def test_linear_weights_and_bias():
    linear = Linear(2, 3)
    assert linear.weights.shape == (2, 3)
    assert linear.bias.shape == (3,)


def test_linear_forward_computation():
    linear = Linear(2, 2)
    linear.weights = Parameter([[1, 2], [3, 4]])
    linear.bias = Parameter([5, 6])
    input_tensor = tensor([[1, 1]])
    output = linear(input_tensor)
    expected_output = tensor([[9, 12]])
    assert output._data == expected_output._data


def test_mse_loss_zero():
    mse_loss = MSELoss()
    input_tensor = tensor([1, 2, 3])
    target_tensor = tensor([1, 2, 3])
    loss = mse_loss(input_tensor, target_tensor)
    assert loss._data[0] == pytest.approx(0.0)
    assert len(loss._data) == 1


def test_mse_loss_nonzero():
    mse_loss = MSELoss()
    input_tensor = tensor([1, 2, 3])
    target_tensor = tensor([4, 5, 6])
    loss = mse_loss(input_tensor, target_tensor)
    expected_loss = (3**2 + 3**2 + 3**2) / 3  # Mean squared error
    assert loss._data[0] == pytest.approx(expected_loss)
    assert len(loss._data) == 1


def test_nested_modules():
    class NestedModule(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(2, 3)
            self.linear2 = Linear(3, 1)

        def forward(self, x):
            x = self.linear1(x)
            return self.linear2(x)

    nested = NestedModule()
    params = list(nested.parameters())
    assert len(params) == 4  # 2 parameters for each Linear module
    assert isinstance(params[0], Parameter)
    assert isinstance(params[1], Parameter)
    assert isinstance(params[2], Parameter)
    assert isinstance(params[3], Parameter)


def test_module_forward_not_implemented():
    module = Module()
    with pytest.raises(NotImplementedError):
        module.forward(Tensor([1, 2, 3]))
