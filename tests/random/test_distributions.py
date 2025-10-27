import math

import pytest
import torch

from jetdl.random import normal, uniform

from ..utils import SEED, generate_shape_ids

# (shape, low, high, use_defaults)
uniform_params = [
    ((10,), 0.0, 1.0),
    ((100,), -1.0, 1.0),
    ((10, 20), 5.0, 10.0),
    ((2, 3, 4), -10.0, -5.0),
    ((100000,), 0.0, 1.0),  # For statistical tests
    ((10,), 0.0, 1.0),  # Test with default params
    ((100000,), 0.0, 1.0),
]


@pytest.mark.parametrize("shape, low, high ", uniform_params, ids=generate_shape_ids)
def test_uniform_properties(shape, low, high):
    j_tensor = uniform(low, high, shape, SEED)

    # 1. Check metadata
    assert j_tensor.shape == shape
    assert j_tensor.size == torch.Size(shape).numel()
    assert j_tensor.ndim == len(shape)

    # 2. Check range
    t_tensor_view = torch.asarray(j_tensor)
    assert torch.all(t_tensor_view >= low)
    assert torch.all(t_tensor_view < high)

    # 3. Check statistics for large tensors
    if j_tensor.size > 1000:
        mean_expected = (low + high) / 2
        var_expected = ((high - low) ** 2) / 12

        mean_actual = t_tensor_view.mean().item()
        var_actual = t_tensor_view.var().item()

        assert math.isclose(mean_actual, mean_expected, rel_tol=0.1, abs_tol=0.01)
        assert math.isclose(var_actual, var_expected, rel_tol=0.1, abs_tol=0.01)


def test_uniform_scalar():
    low, high = 0.0, 5.0
    j_tensor = uniform(low, high)
    assert j_tensor.shape == ()
    assert j_tensor.ndim == 0
    assert j_tensor.size == 1

    # Check if the value is within the range
    value = torch.asarray(j_tensor).item()
    assert low <= value < high


# (shape, mean, std)
normal_params = [
    ((10,), 0.0, 1.0),
    ((100,), 2.0, 5.0),
    ((10, 20), -5.0, 2.0),
    ((100000,), 0.0, 1.0),  # For statistical tests
    ((100000,), 5.0, 2.0),  # For statistical tests
]


@pytest.mark.parametrize("shape, mean, std", normal_params, ids=generate_shape_ids)
def test_normal_properties(shape, mean, std):
    j_tensor = normal(mean, std, shape, SEED)

    # 1. Check metadata
    assert j_tensor.shape == shape
    assert j_tensor.size == torch.Size(shape).numel()
    assert j_tensor.ndim == len(shape)

    # 2. Check statistics for large tensors
    if j_tensor.size > 1000:
        t_tensor_view = torch.asarray(j_tensor)

        mean_expected = mean
        var_expected = std**2

        mean_actual = t_tensor_view.mean().item()
        var_actual = t_tensor_view.var().item()

        assert math.isclose(mean_actual, mean_expected, rel_tol=0.1, abs_tol=0.05)
        assert math.isclose(var_actual, var_expected, rel_tol=0.1, abs_tol=0.05)


def test_normal_scalar():
    mean, std = 10.0, 2.0
    j_tensor = normal(mean, std)
    assert j_tensor.shape == ()
    assert j_tensor.ndim == 0
    assert j_tensor.size == 1

    # Check if the value is reasonable (e.g., within a few std devs)
    value = torch.asarray(j_tensor).item()
    assert mean - 5 * std < value < mean + 5 * std
