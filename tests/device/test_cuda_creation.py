import pytest
import torch

import jetdl

from ..utils import PyTestAsserts

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not jetdl.cuda.is_available(),
    reason="CUDA is not available"
)

init_shapes = [
    (),
    (1,),
    (10,),
    (100,),
    (5,),
    (50,),
    (500,),
    (1, 1),
    (10, 10),
    (100, 100),
    (3, 4),
    (30, 40),
    (300, 400),
    (2, 3, 5),
    (2, 30, 50),
    (2, 300, 500),
    (5, 1),
    (50, 1),
    (500, 1),
]


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("shape", init_shapes)
def test_zeros_cuda(shape, requires_grad):
    """
    Tests the zeros routine on CUDA for various shapes.
    """
    j_tensor = jetdl.zeros(shape, requires_grad=requires_grad, device="cuda")
    t_tensor = torch.zeros(shape)

    assert j_tensor.requires_grad == requires_grad
    assert j_tensor.is_cuda is True

    # Transfer to CPU for comparison
    j_tensor_cpu = j_tensor.cpu()

    assert_object = PyTestAsserts(j_tensor_cpu, t_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("shape", init_shapes)
def test_ones_cuda(shape, requires_grad):
    """
    Tests the ones routine on CUDA for various shapes.
    """
    j_tensor = jetdl.ones(shape, requires_grad=requires_grad, device="cuda")
    t_tensor = torch.ones(shape)

    assert j_tensor.requires_grad == requires_grad
    assert j_tensor.is_cuda is True

    j_tensor_cpu = j_tensor.cpu()

    assert_object = PyTestAsserts(j_tensor_cpu, t_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((), 5.0),
        ((1,), -1.0),
        ((5,), 0.0),
        ((1, 1), 10.0),
        ((3, 4), 3.14),
        ((2, 3, 5), -9.8),
        ((5, 1), 1.0),
    ],
)
def test_fill_cuda(shape, fill_value, requires_grad):
    """
    Tests the fill routine on CUDA for various shapes and fill values.
    """
    j_tensor = jetdl.fill(shape, fill_value, requires_grad=requires_grad, device="cuda")
    t_tensor = torch.full(shape, fill_value)

    assert j_tensor.requires_grad == requires_grad
    assert j_tensor.is_cuda is True

    j_tensor_cpu = j_tensor.cpu()

    assert_object = PyTestAsserts(j_tensor_cpu, t_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


# Random tensor creation on CUDA

random_shapes = [
    (5,),
    (50,),
    (3, 4),
    (30, 40),
    (2, 3, 5),
]


@pytest.mark.parametrize("shape", random_shapes)
def test_uniform_cuda(shape):
    """
    Tests that uniform random tensor creation on CUDA has correct shape and device.
    """
    j_tensor = jetdl.random.uniform(0.0, 1.0, shape, device="cuda")

    assert j_tensor.is_cuda is True
    assert list(j_tensor.shape) == list(shape)


@pytest.mark.parametrize("shape", random_shapes)
def test_normal_cuda(shape):
    """
    Tests that normal random tensor creation on CUDA has correct shape and device.
    """
    j_tensor = jetdl.random.normal(0.0, 1.0, shape, device="cuda")

    assert j_tensor.is_cuda is True
    assert list(j_tensor.shape) == list(shape)


@pytest.mark.parametrize("shape", random_shapes)
def test_rand_cuda(shape):
    """
    Tests that rand tensor creation on CUDA has correct shape and device.
    """
    j_tensor = jetdl.random.rand(*shape, device="cuda")

    assert j_tensor.is_cuda is True
    assert list(j_tensor.shape) == list(shape)