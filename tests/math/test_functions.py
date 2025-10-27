import pytest
import torch

import jetdl

from ..utils import (SEED, PyTestAsserts, generate_random_data,
                     generate_shape_ids)

torch.manual_seed(SEED)

reduction_operation_registry = {
    "sum": (jetdl.sum, torch.sum),
    "mean": (jetdl.mean, torch.mean),
}
reduction_operations_strs = list(reduction_operation_registry.keys())


# (shape, exponent)
shapes_and_exponents = [
    ((10,), 0),
    ((100,), 0),
    ((1000,), 0),
    ((10,), 1),
    ((100,), 1),
    ((1000,), 1),
    ((10,), 2),
    ((100,), 2),
    ((1000,), 2),
    ((3, 4), 3),
    ((30, 40), 3),
    ((2, 3, 4), 0),
    ((2, 30, 40), 0),
    ((2, 3, 4), 1),
    ((2, 30, 40), 1),
    ((2, 3, 4), 2),
    ((2, 30, 40), 2),
    ((2, 3, 4, 5), 3),
    ((2, 30, 40, 50), 3),
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
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize(
    "shape, exponent", shapes_and_exponents, ids=generate_shape_ids
)
def test_tensor_pow_method(shape, exponent):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = j_tensor**exponent
    t_result = t_tensor**exponent

    assert_object = PyTestAsserts(j_result, t_result)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


# (shape)
shapes_for_sqrt = [
    (10,),
    (100,),
    (3, 4),
    (30, 40),
    (2, 3, 4),
    (2, 30, 40),
    (2, 3, 4, 5),
    (2, 30, 40, 50),
]


@pytest.mark.parametrize("shape", shapes_for_sqrt, ids=generate_shape_ids)
def test_sqrt(shape):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = jetdl.sqrt(j_tensor)
    t_result = torch.sqrt(t_tensor)

    assert_object = PyTestAsserts(j_result, t_result)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


def test_sqrt_scalar():
    import math

    j_res = jetdl.sqrt(25.0)
    t_res = math.sqrt(25.0)
    assert j_res == t_res

    j_res_int = jetdl.sqrt(16)
    t_res_int = math.sqrt(16)
    assert j_res_int == t_res_int


# (shape, axis)
shapes_and_axes = [
    ((10,), None),
    ((100,), None),
    ((10,), 0),
    ((3, 4), None),
    ((30, 40), None),
    ((3, 4), 0),
    ((3, 4), 1),
    ((2, 3, 4), None),
    ((20, 3, 40), None),
    ((2, 3, 4), 0),
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
    ((2, 3, 4, 5), None),
    ((2, 30, 4, 50), None),
    ((2, 3, 4, 5), 0),
    ((2, 3, 4, 5), 1),
    ((2, 3, 4, 5), 2),
    ((2, 3, 4, 5), 3),
    # multiple axes
    ((2, 3, 4), (0, 1)),
    ((20, 3, 40), (0, 1)),
    ((2, 3, 4), (0, 2)),
    ((2, 3, 4), (1, 2)),
    ((2, 3, 4, 5), (0, 2)),
    ((2, 30, 4, 50), (0, 2)),
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
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
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


# (from_shape, to_shape, dim, keepdim)
sum_to_shape_params = [
    ((2, 3, 4), (3, 4)),
    ((20, 30, 40), (30, 40)),
    ((2, 3, 4), (2, 1, 4)),
    ((20, 30, 40), (20, 1, 40)),
    ((2, 3, 4), (4,)),
    ((20, 30, 40), (40,)),
    ((5, 2, 3, 4), (3, 4)),
    ((50, 20, 30, 40), (30, 40)),
    ((5, 2, 3, 4), (2, 1, 4)),
    ((50, 20, 30, 40), (20, 1, 40)),
    ((2, 3, 4), (1, 3, 4)),
    ((20, 30, 40), (1, 30, 40)),
    ((2, 3, 4), (2, 1, 4)),
    ((20, 30, 40), (20, 1, 40)),
    ((2, 3, 4), (1, 1, 4)),
    ((20, 30, 40), (1, 1, 40)),
    ((2, 3, 4), (1, 1, 1)),
    ((20, 30, 40), (1, 1, 1)),
    # Special case for summing to (1,)
    ((2, 3, 4), (1,)),
    ((20, 30, 40), (1,)),
]


@pytest.mark.parametrize("from_shape, to_shape", sum_to_shape_params)
def test_sum_to_shape(from_shape, to_shape):
    data = generate_random_data(from_shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = j_tensor.sum_to_shape(to_shape)
    t_result = t_tensor.sum_to_size(to_shape)

    assert_object = PyTestAsserts(j_result, t_result)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("shape, axes", shapes_and_axes, ids=generate_shape_ids)
def test_tensor_sum_method(shape, axes):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = j_tensor.sum(axes=axes)
    t_result = t_tensor.sum(dim=axes)

    assert_object = PyTestAsserts(j_result, t_result)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("shape, axes", shapes_and_axes, ids=generate_shape_ids)
def test_tensor_mean_method(shape, axes):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = j_tensor.mean(axes=axes)
    t_result = t_tensor.mean(dim=axes)

    assert_object = PyTestAsserts(j_result, t_result)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
