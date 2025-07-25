import pytest
import torch

import jetdl

from ..utils import *

torch.manual_seed(SEED)

dot_shapes = [
    ((1), (1)),
    ((5), (5)),
    ((100), (100)),
]

incorrect_dot_shapes = [
    ((), (5)),
    ((4), (3)),
]

matmul_shapes = [
    ((5), (5)),
    ((2, 3), (3)),
    ((3, 2, 4), (4)),
    ((5, 4, 3, 2), (2)),
    ((2), (2, 4)),
    ((3), (4, 3, 2)),
    ((4), (6, 5, 4, 3)),
    ((2, 2), (2, 2)),
    ((3, 2), (2, 4)),
    ((3, 2, 4), (3, 4, 3)),
    ((3, 2, 4), (4, 3)),
    ((1, 2, 3, 4), (1, 4, 3)),
    ((1, 4, 3), (2, 3, 3, 2)),
    ((2, 2, 2, 2, 4), (2, 2, 2, 4, 2)),
    ((1, 2, 3, 4), (4, 3, 2, 4, 2)),
    ((2, 1, 2, 2, 4), (2, 2, 1, 4, 2)),
]


@pytest.mark.parametrize("shape1, shape2", dot_shapes, ids=generate_shape_ids)
def test_dot(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)
    j3 = jetdl.dot(j1, j2)

    t1 = torch.tensor(data1)
    t2 = torch.tensor(data2)
    expected_tensor = torch.dot(t1, t2)

    result_tensor = torch.tensor(j3._data).reshape(j3.shape)
    assert_object = PyTestAsserts(result_tensor, expected_tensor)
    assert assert_object.check_shapes(), assert_object.shapes_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("shape1, shape2", incorrect_dot_shapes, ids=generate_shape_ids)
def test_incorrect_batch_matmul(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)

    with pytest.raises(ValueError) as err:
        j3 = jetdl.dot(j1, j2)
    assert "could not be broadcasted" in str(err.value)


incorrect_matmul_shapes = [
    ((4), (3)),
    ((5, 2), (3)),
    ((3, 2, 4), (5)),
    ((2, 3, 4, 3), (2)),
    ((4), (3, 3)),
    ((2), (4, 3, 3)),
    ((2, 2), (3, 3)),
    ((2, 3), (4, 5)),
    ((1, 2, 3), (4, 5)),
    ((1, 2), (3, 4, 5)),
    ((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
]

incorrect_batch_shapes = [
    ((4, 2, 3), (2, 3, 2)),
    ((1, 2, 3, 4), (3, 4, 5)),
    ((4, 2, 3), (2, 3, 3, 4)),
]


@pytest.mark.parametrize("shape1, shape2", matmul_shapes, ids=generate_shape_ids)
def test_matmul(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)
    j3 = jetdl.matmul(j1, j2)

    t1 = torch.tensor(data1)
    t2 = torch.tensor(data2)
    expected_tensor = torch.matmul(t1, t2)

    result_tensor = torch.tensor(j3._data).reshape(j3.shape)
    assert_object = PyTestAsserts(result_tensor, expected_tensor)
    assert assert_object.check_shapes(), assert_object.shapes_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize(
    "shape1, shape2", incorrect_matmul_shapes, ids=generate_shape_ids
)
def test_incorrect_matmul(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)

    with pytest.raises(ValueError) as err:
        j3 = jetdl.matmul(j1, j2)
    assert "incompatible shapes" in str(err.value)


@pytest.mark.parametrize(
    "shape1, shape2", incorrect_batch_shapes, ids=generate_shape_ids
)
def test_incorrect_batch_matmul(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)

    with pytest.raises(ValueError) as err:
        j3 = jetdl.matmul(j1, j2)
    assert "could not be broadcasted" in str(err.value)
