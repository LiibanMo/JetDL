import pytest
import torch

import jetdl

from ..utils import (SEED, PyTestAsserts, generate_random_data,
                     generate_shape_ids, operation_registry)

torch.manual_seed(SEED)

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not jetdl.cuda.is_available(),
    reason="CUDA is not available"
)

non_broadcast_shapes = [
    ((2), (2)),
    ((20), (20)),
    ((3,), (3,)),
    ((30,), (30,)),
    ((3, 4), (3, 4)),
    ((30, 40), (30, 40)),
    ((4, 3, 2), (4, 3, 2)),
    ((4, 30, 20), (4, 30, 20)),
    ((2, 3, 4, 5), (2, 3, 4, 5)),
    ((2, 30, 40, 50), (2, 30, 40, 50)),
]

broadcast_shapes = [
    ((1), (4)),
    ((1), (40)),
    ((1), (400)),
    ((5), (1)),
    ((50), (1)),
    ((500), (1)),
    ((2, 3), (3)),
    ((20, 30), (30)),
    ((2, 3), (2, 1)),
    ((2, 3, 1, 4), (1, 3, 4, 4)),
]

operations_strs = ["ADD", "SUB", "MUL", "DIV"]


def obtain_cuda_result_tensors(data1, data2, operation: str):
    """Create CUDA tensors, run operation, return result (on CPU) and expected."""
    jetdl_op, torch_op = operation_registry[operation]

    j1 = jetdl.tensor(data1).cuda()
    j2 = jetdl.tensor(data2).cuda()
    j3 = jetdl_op(j1, j2)

    # Verify result is on CUDA
    assert j3.is_cuda is True

    # Transfer back to CPU for comparison
    j3_cpu = j3.cpu()

    t1 = torch.tensor(data1)
    t2 = torch.tensor(data2)
    expected_tensor = torch_op(t1, t2)

    return j3_cpu, expected_tensor


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape1, shape2", non_broadcast_shapes, ids=generate_shape_ids)
def test_non_broadcast_arithmetic_cuda(shape1, shape2, operation):
    data1, data2 = generate_random_data(shape1, shape2)

    j3, expected_tensor = obtain_cuda_result_tensors(data1, data2, operation)

    assert_object = PyTestAsserts(j3, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape1, shape2", broadcast_shapes, ids=generate_shape_ids)
def test_broadcast_arithmetic_cuda(shape1, shape2, operation):
    data1, data2 = generate_random_data(shape1, shape2)

    j3, expected_tensor = obtain_cuda_result_tensors(data1, data2, operation)

    assert_object = PyTestAsserts(j3, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (4, 3, 2)], ids=generate_shape_ids)
def test_scalar_tensor_arithmetic_cuda(shape, operation):
    scalar_data = generate_random_data((), ())[0]
    tensor_data = generate_random_data(shape, shape)[0]

    jetdl_op, torch_op = operation_registry[operation]

    j_scalar = jetdl.tensor(scalar_data).cuda()
    j_tensor = jetdl.tensor(tensor_data).cuda()
    j_result = jetdl_op(j_scalar, j_tensor)

    assert j_result.is_cuda is True
    j_result_cpu = j_result.cpu()

    t_scalar = torch.tensor(scalar_data)
    t_tensor = torch.tensor(tensor_data)
    expected_tensor = torch_op(t_scalar, t_tensor)

    assert_object = PyTestAsserts(j_result_cpu, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("operation", operations_strs)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (4, 3, 2)], ids=generate_shape_ids)
def test_tensor_scalar_arithmetic_cuda(shape, operation):
    tensor_data = generate_random_data(shape, shape)[0]
    scalar_data = generate_random_data((), ())[0]

    jetdl_op, torch_op = operation_registry[operation]

    j_tensor = jetdl.tensor(tensor_data).cuda()
    j_scalar = jetdl.tensor(scalar_data).cuda()
    j_result = jetdl_op(j_tensor, j_scalar)

    assert j_result.is_cuda is True
    j_result_cpu = j_result.cpu()

    t_tensor = torch.tensor(tensor_data)
    t_scalar = torch.tensor(scalar_data)
    expected_tensor = torch_op(t_tensor, t_scalar)

    assert_object = PyTestAsserts(j_result_cpu, expected_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()