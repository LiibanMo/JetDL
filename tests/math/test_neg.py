import pytest
import torch

import jetdl

from ..utils import SEED, PyTestAsserts, generate_random_data

torch.manual_seed(SEED)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (5,),
        (3, 4),
        (2, 3, 5),
    ],
)
def test_neg(shape):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    j_result = -j_tensor
    t_result = -t_tensor

    assert_object = PyTestAsserts(j_result, t_result)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
