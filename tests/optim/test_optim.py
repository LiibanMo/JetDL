import pytest
import torch

import jetdl
import jetdl.nn as nn
from jetdl.optim import Optimizer

from ..utils import SEED

torch.manual_seed(SEED)


def test_optimizer_init_with_params():
    """
    Tests that the optimizer correctly holds the parameters from a model.
    """
    model = nn.Linear(10, 2)
    params = list(model.parameters())
    optim = Optimizer(params)
    assert optim.params == params
    assert len(optim.params) == 2  # weight and bias


def test_optimizer_init_empty_params():
    """
    Tests that the optimizer can be initialized with an empty parameter list.
    """
    optim = Optimizer([])
    assert optim.params == []


def test_optimizer_step_not_implemented():
    """
    Tests that the base Optimizer's step method raises NotImplementedError.
    """
    model = nn.Linear(10, 2)
    optim = Optimizer(model.parameters())
    with pytest.raises(NotImplementedError, match="Optimizer.step not implemented."):
        optim.step()


def test_optimizer_zero_grad():
    """
    Tests that zero_grad sets the gradients of all parameters to zero.
    """
    # A single epoch to obtain gradients

    model = nn.Linear(10, 2)

    optim = Optimizer(model.parameters())
    input_tensor = jetdl.ones((1, 10))
    output = model(input_tensor)
    
    loss = jetdl.mean(output)
    loss.backward()

    # Simulate some gradients
    for p in optim.params:
        assert p.grad is not None
        # Make sure grad is not all zeros before zero_grad
        grad_torch_before = torch.asarray(p.grad).reshape(p.grad.shape)
        assert not torch.all(grad_torch_before == 0)

    # Zero the gradients
    optim.zero_grad()

    # Check if gradients are zero
    for p in optim.params:
       assert p.grad is None 


def test_optimizer_zero_grad_no_params():
    """
    Tests that zero_grad runs without error with no parameters.
    """
    optim = Optimizer([])
    try:
        optim.zero_grad()
    except Exception as e:
        pytest.fail(f"Optimizer.zero_grad() with no parameters raised an exception: {e}")
