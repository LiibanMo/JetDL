import pytest
import torch

import jetdl
import jetdl.nn as nn
from jetdl.optim import SGD

from ..utils import SEED, PyTestAsserts

torch.manual_seed(SEED)


def test_sgd_init():
    """
    Tests the initialization of the SGD optimizer.
    """
    model = nn.Linear(10, 2)
    params = list(model.parameters())
    lr = 0.1
    optim = SGD(params, lr=lr)
    assert optim.params == params
    assert optim.lr == lr


def test_sgd_step():
    """
    Tests that the SGD step updates parameters correctly.
    """
    # --- Setup JetDL model, loss and optimizer ---
    j_model = nn.Linear(10, 5)
    j_criterion = nn.MSELoss()
    j_optim = SGD(j_model.parameters(), lr=0.1)

    # --- Setup PyTorch model, loss and optimizer ---
    t_model = torch.nn.Linear(10, 5)
    t_criterion = torch.nn.MSELoss()
    t_optim = torch.optim.SGD(t_model.parameters(), lr=0.1)

    # --- Synchronize model parameters ---
    with torch.no_grad():
        t_model.weight.copy_(torch.asarray(j_model.weight).reshape(j_model.weight.shape))
        t_model.bias.copy_(torch.asarray(j_model.bias).reshape(j_model.bias.shape))

    # --- Synthesise data and target
    input_data = torch.rand(10, dtype=torch.float32, requires_grad=True)
    target_data = torch.arange(5, dtype=torch.float32, requires_grad=True)

    # --- Forward and backward pass to get gradients ---
    
    # JetDL
    j_input = jetdl.tensor(input_data.tolist())
    j_output = j_model(j_input)
    j_loss = j_criterion(j_output, target_data.tolist())
    j_loss.backward()

    # PyTorch
    t_input = input_data.clone()
    t_output = t_model(t_input)
    t_loss = t_criterion(t_output, target_data)
    t_loss.backward()

    # --- Perform optimization step ---
    j_optim.step()
    t_optim.step()

    # --- Compare updated parameters ---
    j_params = list(j_model.parameters())
    t_params = list(t_model.parameters())

    for i in range(len(j_params)):
        j_param = j_params[i]
        t_param = t_params[i]
        assert_obj = PyTestAsserts(j_param, t_param)
        assert assert_obj.check_results(err=1e-5), assert_obj.results_error_output()
