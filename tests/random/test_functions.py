import math

import pytest
import torch

import jetdl
from jetdl.random import rand

from ..utils import SEED, generate_shape_ids

rand_shapes = [
    ((1,)),
    ((2, 3)),
    ((4, 2, 5)),
    ((3, 1, 4, 1)),
]


@pytest.mark.parametrize("shape", rand_shapes, ids=generate_shape_ids)
def test_rand_properties(shape):
    j_tensor = jetdl.rand(*shape, seed=SEED)

    # 1. Check metadata
    assert j_tensor.shape == shape
    assert j_tensor.size == torch.Size(shape).numel()
    assert j_tensor.ndim == len(shape)

    # 2. Check range
    t_tensor_view = torch.asarray(j_tensor)
    assert torch.all(t_tensor_view >= 0)
    assert torch.all(t_tensor_view < 1)
    # 3. Check statistics for large tensors
    if j_tensor.size > 1000:
        mean_expected = 1 / 2
        var_expected = 1 / 12

        mean_actual = t_tensor_view.mean().item()
        var_actual = t_tensor_view.var().item()

        assert math.isclose(mean_actual, mean_expected, rel_tol=0.1, abs_tol=0.01)
        assert math.isclose(var_actual, var_expected, rel_tol=0.1, abs_tol=0.01)
