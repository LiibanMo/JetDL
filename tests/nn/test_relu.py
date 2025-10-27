import pytest
import torch

import jetdl
import jetdl.nn as nn

from ..utils import PyTestAsserts, generate_random_data, generate_shape_ids

# nn.ReLU


def test_relu_initialization():
    """
    Tests that a nn.ReLU has no parameters.
    """
    relu_layer = nn.ReLU()
    assert len(list(relu_layer.parameters())) == 0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 10, 5),
        (8, 20, 10),
        (1, 1, 1, 1),
        (3, 4, 5, 6),
    ],
    ids=generate_shape_ids,
)
def test_relu_forward(shape):
    j_relu = nn.ReLU()
    t_relu = torch.nn.ReLU()

    # Generate data with negative values to test ReLU properly
    input_data = generate_random_data(shape)
    j_input = jetdl.tensor(input_data)
    t_input = torch.tensor(input_data, dtype=torch.float32)

    j_output = j_relu(j_input)
    t_output = t_relu(t_input)

    assert_object = PyTestAsserts(j_output, t_output)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
