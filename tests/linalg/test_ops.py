import pytest
import torch

import jetdl

from ..utils import *

torch.manual_seed(SEED)

non_broadcast_shapes = [
    ((2), (2)),
    ((3, 4), (3, 4)),
    ((4, 3, 2), (4, 3, 2)),
    ((2, 3, 4, 5), (2, 3, 4, 5)),
]

broadcast_shapes = [
    ((1), (4)),
    ((5), (1)),
    ((2, 3), (3)),
    ((2, 3), (2)),
    ((2, 3, 1, 4), (1, 3, 4, 4)),
]


@pytest.mark.parametrize("shape1, shape2", non_broadcast_shapes, ids=generate_shape_ids)
def test_non_broadcast_addition(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)
    j3 = jetdl.add(j1, j2)

    t1 = torch.tensor(data1)
    t2 = torch.tensor(data2)
    expected_tensor = t1 + t2

    result_tensor = torch.tensor(j3._data).reshape(j3.shape)
    assert_object = PyTestAsserts(result_tensor, expected_tensor)
    assert assert_object.check_shapes(), assert_object.shapes_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()
