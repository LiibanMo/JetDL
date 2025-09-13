import pytest
import torch

import jetdl

from ..utils import PyTestAsserts


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (5,),
        (1, 1),
        (3, 4),
        (2, 3, 5),
        (5, 1),
    ],
)
def test_ones(shape):
    """
    Tests the ones routine for various shapes.
    """
    j_tensor = jetdl.ones(shape)
    t_tensor = torch.ones(shape)

    assert_object = PyTestAsserts(j_tensor, t_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
