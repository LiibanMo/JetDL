import pytest
import torch

import jetdl

from ..utils import *

torch.manual_seed(SEED)

# (shape, axis)
shapes_and_axes = [
    ((10,), None),
    ((10,), 0),
    ((3, 4), None),
    ((3, 4), 0),
    ((3, 4), 1),
    ((2, 3, 4), None),
    ((2, 3, 4), 0),
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
    ((2, 3, 4, 5), None),
    ((2, 3, 4, 5), 0),
    ((2, 3, 4, 5), 1),
    ((2, 3, 4, 5), 2),
    ((2, 3, 4, 5), 3),
]

@pytest.mark.parametrize("shape, axis", shapes_and_axes, ids=generate_shape_ids)
def test_sum(shape, axis):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = jetdl.sum(j_tensor, axes=axis)
    t_result = torch.sum(t_tensor, dim=axis)

    assert_object = PyTestAsserts(j_result, t_result)
    assert assert_object.check_basic_metadata(), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


# (shape, axes)
shapes_and_multiple_axes = [
    ((2, 3, 4), (0, 1)),
    ((2, 3, 4), (0, 2)),
    ((2, 3, 4), (1, 2)),
    ((2, 3, 4, 5), (0, 2)),
    ((2, 3, 4, 5), (1, 3)),
    ((2, 3, 4, 5), (0, 1, 2)),
    ((2, 3, 4, 5), (1, 2, 3)),
    ((2, 3, 4, 5), (0, 1, 2, 3)),
]


@pytest.mark.parametrize("shape, axes", shapes_and_multiple_axes, ids=generate_shape_ids)
def test_sum_multiple_axes(shape, axes):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = jetdl.sum(j_tensor, axes=axes)
    t_result = torch.sum(t_tensor, dim=axes)

    assert_object = PyTestAsserts(j_result, t_result)
    assert assert_object.check_basic_metadata(), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


# (shape, axes)
shapes_and_oob_axes = [ # out of bounds
    ((2, 3, 4), 3),
    ((2, 3, 4), -4),
    ((2, 3, 4), (0, 3)),
    ((2, 3, 4), (0, -4)),
    ((2, 3, 4, 5), (0, 1, 4)),
    ((2, 3, 4, 5), (0, 1, -5)),
]


@pytest.mark.parametrize("shape, axes", shapes_and_oob_axes, ids=generate_shape_ids)
def test_sum_invalid_axes(shape, axes):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)

    with pytest.raises(IndexError):
        jetdl.sum(j_tensor, axes=axes)


shapes_and_oob_axes = [ # duplicates
    ((2, 3, 4), (0, 0)),
    ((2, 3, 4), (-1, 2)),
    ((2, 3, 4, 5), (0, -1, 3)),
    ((2, 3, 4, 5), (-3, 1, 0)),
]


@pytest.mark.parametrize("shape, axes", shapes_and_oob_axes, ids=generate_shape_ids)
def test_sum_invalid_axes(shape, axes):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)

    with pytest.raises(RuntimeError):
        jetdl.sum(j_tensor, axes=axes)