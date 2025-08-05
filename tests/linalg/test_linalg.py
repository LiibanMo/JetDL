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

@pytest.mark.parametrize("shape", [
    (5,),
    (2, 3),
    (1, 2, 3),
    (1, 2, 3, 4),
    (1, 2, 3, 4, 5),
    (6, 5, 4, 3, 2, 1),
])
def test_transpose(shape):
    data = generate_random_data(shape)
    flattened_data = torch.flatten(torch.tensor(data)).tolist()

    j_tensor = jetdl.tensor(data)
    jetdl_transposed = j_tensor.T
    
    expected_shape = tuple(reversed(shape))

    assert jetdl_transposed.shape == expected_shape, f"Expected shapes to match: (jetdl) {jetdl_transposed.shape} vs (actual) {expected_shape}"
    assert jetdl_transposed._data == pytest.approx(flattened_data, ERR), f"Expected data to be close: {jetdl_transposed.data} vs {flattened_data}"
    assert jetdl_transposed.is_contiguous == False

@pytest.mark.parametrize("shape", [
    (2, 3),
    (1, 2, 3),
    (1, 2, 3, 4),
    (1, 2, 3, 4, 5),
    (6, 5, 4, 3, 2, 1),
])
def test_matrix_transpose(shape):
    data = generate_random_data(shape)
    flattened_data = torch.flatten(torch.tensor(data)).tolist()

    j_tensor = jetdl.tensor(data)
    jetdl_matrix_transposed = j_tensor.mT

    expected_shape = list(shape).copy()
    expected_shape[-1], expected_shape[-2] = expected_shape[-2], expected_shape[-1]

    assert jetdl_matrix_transposed.shape == tuple(expected_shape), f"Expected shapes to match: (jetdl) {jetdl_matrix_transposed.shape} vs (actual) {tuple(expected_shape)}"
    assert jetdl_matrix_transposed._data == pytest.approx(flattened_data, ERR), f"Expected data to be close: {jetdl_matrix_transposed.data} vs {flattened_data}"
    assert jetdl_matrix_transposed.is_contiguous == False

@pytest.mark.parametrize("shape", [
    (5),
    (5,),
])
def test_incorrect_matrix_transpose(shape):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)
    
    assert j_tensor.shape == t_tensor.shape
    assert j_tensor.ndim == t_tensor.ndim
    with pytest.raises(RuntimeError) as err:
        _ = j_tensor.mT
    assert "only supports matrices or batches of matrices" in str(err.value)