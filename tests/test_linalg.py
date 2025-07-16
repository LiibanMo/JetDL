import pytest
import jetdl
import torch

torch.manual_seed(123)

def generate_random_data(shape1, shape2):
    return torch.rand(shape1).tolist(), torch.rand(shape2).tolist()

def generate_shape_ids(shapes):
    return f" {shapes} "

matmul_shapes = [
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

incorrect_matmul_shapes = [
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
    t3 = torch.matmul(t1, t2)

    expected_result = torch.tensor(j3._data).reshape(j3.shape)
    assert j3.shape == t3.shape, f"Expected shapes to match: {j3.shape} vs {t3.shape}"
    assert torch.allclose(expected_result, t3), f"Expected tensors to be close: {expected_result} vs {t3}"

@pytest.mark.parametrize("shape1, shape2", incorrect_matmul_shapes, ids=generate_shape_ids)
def test_incorrect_matmul(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)

    with pytest.raises(ValueError) as err:
        j3 = jetdl.matmul(j1, j2)
    assert "incompatible shapes" in str(err.value)

@pytest.mark.parametrize("shape1, shape2", incorrect_batch_shapes, ids=generate_shape_ids)
def test_incorrect_batch_matmul(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)

    with pytest.raises(ValueError) as err:
        j3 = jetdl.matmul(j1, j2)
    assert "could not be broadcasted" in str(err.value)