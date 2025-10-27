import pytest
import torch

import jetdl
from jetdl.nn import Parameter

from ..utils import PyTestAsserts, generate_random_data, generate_shape_ids

# nn.Parameter

@pytest.mark.parametrize("shape", [
    ([10, 5]),
    ([8, 20]),
    ([1, 1, 1, 1]),
    ([3, 4, 5, 6]),
], ids=generate_shape_ids)
def test_parameter_he_initialization(shape):
    param = Parameter(shape, init_type="he")

    assert param.shape == tuple(shape)
    assert param.ndim == len(shape)
    assert param.requires_grad is True

    # He initialization bounds check (approximate)
    # n_in = shape[0]
    # bound = (6 / n_in)**0.5
    # Check if values are within a reasonable range for He initialization
    # This is a loose check as exact values depend on the random seed
    # Need to implement min & max first:
    # assert param.min() >= -bound * 5 # Loosen bound for random values
    # assert param.max() <= bound * 5 # Loosen bound for random values

@pytest.mark.parametrize("shape", [
    ([10, 5]),
    ([8, 20]),
    ([1, 1, 1, 1]),
    ([3, 4, 5, 6]),
], ids=generate_shape_ids)
def test_parameter_zero_initialization(shape):
    param = Parameter(shape, init_type="zero")

    assert param.shape == tuple(shape)
    assert param.ndim == len(shape)
    assert param.requires_grad is True
    # Need to implement inequality checks first
    # assert (param == 0).all()

def test_parameter_not_implemented_init_type():
    with pytest.raises(NotImplementedError, match="init type 'invalid' not implemented for Parameter"):
        Parameter([2, 2], init_type="invalid")

