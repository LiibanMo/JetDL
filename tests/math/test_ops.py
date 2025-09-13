import pytest
import torch

import jetdl

from ..utils import *

torch.manual_seed(SEED)

non_broadcast_shapes = [
    ((2), (2)),
    ((3,), (3,)),
    ((3, 4), (3, 4)),
    ((4, 3, 2), (4, 3, 2)),
    ((2, 3, 4, 5), (2, 3, 4, 5)),
]

broadcast_shapes = [
    ((1), (4)),
    ((5), (1)),
    ((2, 3), (3)),
    ((2, 3), (2, 1)),
    ((2, 3, 1, 4), (1, 3, 4, 4)),
]

operations_strs = ["ADD", "SUB", "MUL", "DIV"]


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape1, shape2", non_broadcast_shapes, ids=generate_shape_ids)
def test_non_broadcast_addition(shape1, shape2, operation):
    data1, data2 = generate_random_data(shape1, shape2)

    j3, expected_tensor = obtain_result_tensors(data1, data2, operation)

    assert_object = PyTestAsserts(j3, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape1, shape2", broadcast_shapes, ids=generate_shape_ids)
def test_broadcast_addition(shape1, shape2, operation):
    data1, data2 = generate_random_data(shape1, shape2)

    j3, expected_tensor = obtain_result_tensors(data1, data2, operation)

    assert_object = PyTestAsserts(j3, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (4, 3, 2)], ids=generate_shape_ids)
def test_scalar_tensor_operations(shape, operation):
    scalar_data = generate_random_data((), ())[0]  # Get a single scalar value
    tensor_data = generate_random_data(shape, shape)[0]

    jetdl_op, torch_op = operation_registry[operation]

    j_scalar = jetdl.tensor(scalar_data)
    j_tensor = jetdl.tensor(tensor_data)
    j_result = jetdl_op(j_scalar, j_tensor)
    print(f"j_result = {j_result._data}")
    print(f"j_result_shape = {j_result.shape}")

    t_scalar = torch.tensor(scalar_data)
    t_tensor = torch.tensor(tensor_data)
    expected_tensor = torch_op(t_scalar, t_tensor)
    print(f"expected_tensor = {expected_tensor}")
    print(f"expected_tensor = {expected_tensor.shape}")

    assert_object = PyTestAsserts(j_result, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (4, 3, 2)], ids=generate_shape_ids)
def test_tensor_scalar_operations(shape, operation):
    tensor_data = generate_random_data(shape, shape)[0]
    scalar_data = generate_random_data((), ())[0]  # Get a single scalar value

    jetdl_op, torch_op = operation_registry[operation]

    j_tensor = jetdl.tensor(tensor_data)
    j_scalar = jetdl.tensor(scalar_data)
    j_result = jetdl_op(j_tensor, j_scalar)

    t_tensor = torch.tensor(tensor_data)
    t_scalar = torch.tensor(scalar_data)
    expected_tensor = torch_op(t_tensor, t_scalar)

    assert_object = PyTestAsserts(j_result, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
