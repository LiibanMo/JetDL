import pytest
import torch

import jetdl
import jetdl.nn as nn

from ..utils import PyTestAsserts, generate_random_data, generate_shape_ids

# nn.MSELoss

def test_mseloss_initialization():
    """
    Tests that a nn.MSELoss has no parameters.
    """
    mseloss_layer = nn.MSELoss()
    assert len(list(mseloss_layer.parameters())) == 0

@pytest.mark.parametrize("shape", [
    (1, 10, 5),
    (8, 20, 10),
    (1, 1, 1, 1),
    (3, 4, 5, 6),
], ids=generate_shape_ids)
def test_mseloss_forward(shape):
    j_mseloss = nn.MSELoss()
    t_mseloss = torch.nn.MSELoss()

    # Generate random data for y_true and y_pred
    y_true_data = generate_random_data(shape)
    y_pred_data = generate_random_data(shape)

    j_y_true = jetdl.tensor(y_true_data)
    j_y_pred = jetdl.tensor(y_pred_data)
    t_y_true = torch.tensor(y_true_data, dtype=torch.float32)
    t_y_pred = torch.tensor(y_pred_data, dtype=torch.float32)

    j_output = j_mseloss(j_y_true, j_y_pred)
    t_output = t_mseloss(t_y_true, t_y_pred)

    assert_object = PyTestAsserts(j_output, t_output)
    assert assert_object.check_basic_metadata(), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
