import pytest
import torch

import jetdl

from ..utils import ERR, SEED, PyTestAsserts, generate_random_data, generate_shape_ids

torch.manual_seed(SEED)

dot_shapes = [
    ((1), (1)),
    ((5), (5)),
    ((50), (50)),
    ((100), (100)),
]

incorrect_dot_shapes = [
    ((), (5)),
    ((4), (3)),
]

matmul_shapes = [
    ((5), (5)),
    ((50), (50)),
    ((2, 3), (3)),
    ((20, 30), (30)),
    ((3, 2, 4), (4)),
    ((3, 20, 40), (40)),
    ((5, 4, 3, 2), (2)),
    ((2), (2, 4)),
    ((3), (4, 3, 2)),
    ((4), (6, 5, 4, 3)),
    ((2, 2), (2, 2)),
    ((20, 20), (20, 20)),
    ((1024, 1024), (1024, 1024)),
    ((3, 2), (2, 4)),
    ((30, 20), (20, 40)),
    ((3, 2, 4), (3, 4, 3)),
    ((3, 20, 40), (3, 40, 30)),
    ((3, 224, 224), (3, 224, 224)),
    ((4, 1024, 1024), (4, 1024, 1024)),
    ((3, 2, 4), (4, 3)),
    ((3, 20, 40), (40, 30)),
    ((1, 2, 3, 4), (1, 4, 3)),
    ((1, 2, 30, 40), (1, 40, 30)),
    ((1, 4, 3), (2, 3, 3, 2)),
    ((1, 40, 30), (2, 3, 30, 20)),
    ((2, 2, 2, 2, 4), (2, 2, 2, 4, 2)),
    ((1, 2, 3, 4), (4, 3, 2, 4, 2)),
    ((1, 3, 224, 224), (4, 3, 3, 224, 224)),
    ((2, 1, 2, 1024, 1024), (2, 2, 1, 1024, 1024)),
]


@pytest.mark.parametrize("shape1, shape2", dot_shapes, ids=generate_shape_ids)
def test_dot(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)
    j3 = jetdl.dot(j1, j2)

    t1 = torch.tensor(data1)
    t2 = torch.tensor(data2)
    t3 = torch.dot(t1, t2)

    assert_object = PyTestAsserts(j3, t3)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("shape1, shape2", incorrect_dot_shapes, ids=generate_shape_ids)
def test_incorrect_batch_matmul(shape1, shape2):
    data1, data2 = generate_random_data(shape1, shape2)

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)

    with pytest.raises(ValueError) as err:
        _ = jetdl.dot(j1, j2)
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
    t3 = torch.matmul(t1, t2)

    # Scale tolerance with inner dimension K to account for floating-point
    # error accumulation in large matrix multiplications. Different BLAS
    # implementations (Accelerate vs OpenBLAS vs MKL) may produce slightly
    # different results, especially on Windows.
    K = shape1[-1] if isinstance(shape1, tuple) else shape1
    N = int(len(shape1))
    scaled_rtol = ERR * max(1, K**N)
    scaled_atol = 1e-1 * max(1, K**N)

    assert_object = PyTestAsserts(j3, t3)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(
        rtol=scaled_rtol, atol=scaled_atol
    ), assert_object.results_error_output()


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


@pytest.mark.parametrize(
    "shape",
    [
        (5,),
        (2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (6, 5, 4, 3, 2, 1),
    ],
)
def test_transpose(shape):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    jetdl_transposed = j_tensor.T

    t_tensor = torch.tensor(data)
    torch_transposed = t_tensor.permute(*torch.arange(t_tensor.ndim - 1, -1, -1))

    assert_object = PyTestAsserts(jetdl_transposed, torch_transposed)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (6, 5, 4, 3, 2, 1),
    ],
)
def test_matrix_transpose(shape):
    data = generate_random_data(shape)

    j_tensor = jetdl.tensor(data)
    jetdl_matrix_transposed = j_tensor.mT

    t_tensor = torch.tensor(data)
    torch_matrix_transposed = t_tensor.mT

    assert_object = PyTestAsserts(jetdl_matrix_transposed, torch_matrix_transposed)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5),
        (5,),
    ],
)
def test_incorrect_matrix_transpose(shape):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    assert_object = PyTestAsserts(j_tensor, t_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()

    with pytest.raises(RuntimeError) as err:
        _ = j_tensor.mT
    assert "only supports matrices or batches of matrices" in str(err.value)
