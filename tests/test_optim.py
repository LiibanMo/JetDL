import pytest

from jetdl import ones
from jetdl.nn import Parameter
from jetdl.optim import SGD

@pytest.fixture
def mock_parameter():
    # Mock tensor for testing
    param = Parameter([1.0, 2.0, 3.0])
    param.grad = 1.0
    return param

@pytest.fixture
def mock_grad():
    # Mock gradient for testing
    return [0.1, 0.2, 0.3]

def test_sgd_step(mock_parameter, mock_grad):
    lr = 0.01
    optimizer = SGD(lr=lr, params=mock_parameter)
    optimizer.step()
    updated_param = optimizer.params[0]
    expected_step = lr * mock_parameter.grad
    # Expected tensor after SGD step
    expected_param = Parameter([1.0 - expected_step , 2.0 - expected_step, 3.0 - expected_step])
    assert updated_param._data == pytest.approx(expected_param._data)

def test_sgd_zero_grad(mock_parameter, mock_grad):
    lr = 0.01
    optimizer = SGD(lr=lr, params=mock_parameter)
    optimizer.step()
    optimizer.zero_grad()
    
    # After zero_grad, gradients should be reset to zero
    assert optimizer.params[0].grad == 0.0
    assert len(optimizer.params) == 1