import pytest
import torch

import jetdl

from ..utils import PyTestAsserts

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
def test_zeros(shape, requires_grad):
    """
    Tests the zeros routine for various shapes.
    """
    j_tensor = jetdl.zeros(shape, requires_grad=requires_grad)
    t_tensor = torch.zeros(shape)

    assert j_tensor.requires_grad == requires_grad

    assert_object = PyTestAsserts(j_tensor, t_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("shape", init_shapes)
def test_ones(shape, requires_grad):
    """
    Tests the ones routine for various shapes.
    """
    j_tensor = jetdl.ones(shape, requires_grad=requires_grad)
    t_tensor = torch.ones(shape)

    assert j_tensor.requires_grad == requires_grad

    assert_object = PyTestAsserts(j_tensor, t_tensor)
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
def test_fill(shape, fill_value, requires_grad):
    """
    Tests the fill routine for various shapes and fill values.
    """
    j_tensor = jetdl.fill(shape, fill_value, requires_grad=requires_grad)
    t_tensor = torch.full(shape, fill_value)

    assert j_tensor.requires_grad == requires_grad

    assert_object = PyTestAsserts(j_tensor, t_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
