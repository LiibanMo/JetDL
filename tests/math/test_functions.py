import pytest
import torch

import jetdl

from ..utils import (
    SEED,
    PyTestAsserts,
    generate_random_data,
    generate_shape_ids,
)

torch.manual_seed(SEED)

reduction_operation_registry = {
    "sum": (jetdl.sum, torch.sum),
    "mean": (jetdl.mean, torch.mean),
}
reduction_operations_strs = list(reduction_operation_registry.keys())


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
    # multiple axes
    ((2, 3, 4), (0, 1)),
    ((2, 3, 4), (0, 2)),
    ((2, 3, 4), (1, 2)),
    ((2, 3, 4, 5), (0, 2)),
    ((2, 3, 4, 5), (1, 3)),
    ((2, 3, 4, 5), (0, 1, 2)),
    ((2, 3, 4, 5), (1, 2, 3)),
    ((2, 3, 4, 5), (0, 1, 2, 3)),
]


@pytest.mark.parametrize("operation", reduction_operations_strs)
@pytest.mark.parametrize("shape, axes", shapes_and_axes, ids=generate_shape_ids)
def test_reduction(operation, shape, axes):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    jetdl_op, torch_op = reduction_operation_registry[operation]

    j_result = jetdl_op(j_tensor, axes=axes)
    t_result = torch_op(t_tensor, dim=axes)

    assert_object = PyTestAsserts(j_result, t_result)
    assert assert_object.check_basic_metadata(), (
        assert_object.basic_metadata_error_output()
    )
    assert assert_object.check_results(), assert_object.results_error_output()


# (shape, axes)
shapes_and_oob_axes = [  # out of bounds
    ((2, 3, 4), 3),
    ((2, 3, 4), -4),
    ((2, 3, 4), (0, 3)),
    ((2, 3, 4), (0, -4)),
    ((2, 3, 4, 5), (0, 1, 4)),
    ((2, 3, 4, 5), (0, 1, -5)),
]


@pytest.mark.parametrize("operation", reduction_operations_strs)
@pytest.mark.parametrize("shape, axes", shapes_and_oob_axes, ids=generate_shape_ids)
def test_reduction_invalid_axes_oob(operation, shape, axes):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    jetdl_op, _ = reduction_operation_registry[operation]

    with pytest.raises(IndexError):
        jetdl_op(j_tensor, axes=axes)


shapes_and_dup_axes = [  # duplicates
    ((2, 3, 4), (0, 0)),
    ((2, 3, 4), (-1, 2)),
    ((2, 3, 4, 5), (0, -1, 3)),
    ((2, 3, 4, 5), (-3, 1, 0)),
]


@pytest.mark.parametrize("operation", reduction_operations_strs)
@pytest.mark.parametrize("shape, axes", shapes_and_dup_axes, ids=generate_shape_ids)
def test_reduction_invalid_axes_dups(operation, shape, axes):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    jetdl_op, _ = reduction_operation_registry[operation]

    with pytest.raises(RuntimeError):
        jetdl_op(j_tensor, axes=axes)


# (shape, exponent)
shapes_and_exponents = [
    ((10,), 0),
    ((10,), 1),
    ((10,), 2),
    ((3, 4), 3),
    ((2, 3, 4), 0),
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
    ((2, 3, 4, 5), 3),
]


@pytest.mark.parametrize(
    "shape, exponent", shapes_and_exponents, ids=generate_shape_ids
)
def test_pow(shape, exponent):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = jetdl.pow(j_tensor, exponent)
    t_result = torch.pow(t_tensor, exponent)

    assert_object = PyTestAsserts(j_result, t_result)
    assert assert_object.check_basic_metadata(), (
        assert_object.basic_metadata_error_output()
    )
    assert assert_object.check_results(), assert_object.results_error_output()
