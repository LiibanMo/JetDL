import pytest
import torch

import jetdl

from ..utils import SEED, PyTestAsserts, generate_random_data

torch.manual_seed(SEED)


# --- Contiguous ---


def test_contiguous():
    data = generate_random_data((2, 3))
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    # Transpose to make it non-contiguous
    j_transposed = j_tensor.T
    t_transposed = t_tensor.T

    assert not j_transposed.is_contiguous
    assert not t_transposed.is_contiguous()

    # Make it contiguous
    j_contiguous = jetdl.contiguous(j_transposed)
    t_contiguous = t_transposed.contiguous()

    assert j_contiguous.is_contiguous
    assert t_contiguous.is_contiguous()

    assert_object = PyTestAsserts(j_contiguous, t_contiguous)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


def test_contiguous_on_contiguous():
    data = generate_random_data((2, 3))
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    assert j_tensor.is_contiguous
    assert t_tensor.is_contiguous()

    j_contiguous = jetdl.contiguous(j_tensor)
    t_contiguous = t_tensor.contiguous()

    assert_object = PyTestAsserts(j_contiguous, t_contiguous)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


# --- Reshape ---

reshape_params = [
    ((2, 3), (6,)),
    ((6,), (2, 3)),
    ((2, 3, 4), (24,)),
    ((24,), (2, 3, 4)),
    ((2, 3, 4), (6, 4)),
]


@pytest.mark.parametrize("shape, new_shape", reshape_params)
def test_reshape(shape, new_shape):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    j_reshaped = jetdl.reshape(j_tensor, new_shape)
    t_reshaped = torch.reshape(t_tensor, new_shape)

    assert_object = PyTestAsserts(j_reshaped, t_reshaped)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("shape, new_shape", reshape_params)
def test_tensor_reshape_method(shape, new_shape):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    j_reshaped = j_tensor.reshape(new_shape)
    t_reshaped = t_tensor.reshape(new_shape)

    assert_object = PyTestAsserts(j_reshaped, t_reshaped)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


reshape_error_params = [
    ((2, 3), (5,)),
    ((2, 3), (-2, 3)),
    ((2, 3), (2, -3)),
    ((6,), (2, 4)),
    ((6,), (-6,)),
    ((2, 3, 4), (6,)),
    ((2, 3, 4), (2, 3, 5)),
    ((2, 3, 4), (2, -3, 4)),
]


@pytest.mark.parametrize("shape, new_shape", reshape_error_params)
def test_reshape_errors(shape, new_shape):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)

    with pytest.raises(RuntimeError):
        jetdl.reshape(j_tensor, new_shape)


# --- Squeeze ---

squeeze_params = [
    ((1, 3, 1), None),
    ((1, 3, 1), 0),
    ((1, 3, 1), 2),
    ((1, 3, 1), (0, 2)),
    ((2, 1, 3), 1),
    ((2, 3, 1), None),
    ((2, 3), None),
]


@pytest.mark.parametrize("shape, axes", squeeze_params)
def test_squeeze(shape, axes):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    if axes is None:
        j_squeezed = jetdl.squeeze(j_tensor, axes=None)
        t_squeezed = torch.squeeze(t_tensor)
    else:
        j_squeezed = jetdl.squeeze(j_tensor, axes=axes)
        t_squeezed = torch.squeeze(t_tensor, dim=axes)

    assert_object = PyTestAsserts(j_squeezed, t_squeezed)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("shape, axes", squeeze_params)
def test_tensor_squeeze_method(shape, axes):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    if axes is None:
        j_squeezed = j_tensor.squeeze()
        t_squeezed = torch.squeeze(t_tensor)
    else:
        j_squeezed = j_tensor.squeeze(axes)
        t_squeezed = torch.squeeze(t_tensor, dim=axes)

    assert_object = PyTestAsserts(j_squeezed, t_squeezed)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


# --- Unsqueeze ---

unsqueeze_params = [
    ((3, 4), 0),
    ((3, 4), 1),
    ((3, 4), -1),
    ((5,), 0),
    ((5,), -1),
]


@pytest.mark.parametrize("shape, axis", unsqueeze_params)
def test_unsqueeze(shape, axis):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    j_unsqueezed = jetdl.unsqueeze(j_tensor, axis=axis)
    t_unsqueezed = torch.unsqueeze(t_tensor, dim=axis)

    assert_object = PyTestAsserts(j_unsqueezed, t_unsqueezed)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize("shape, axis", unsqueeze_params)
def test_tensor_unsqueeze_method(shape, axis):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    j_unsqueezed = j_tensor.unsqueeze(axis)
    t_unsqueezed = torch.unsqueeze(t_tensor, dim=axis)

    assert_object = PyTestAsserts(j_unsqueezed, t_unsqueezed)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


unsqueeze_error_params = [
    ((3, 4), 3),
    ((3, 4), -3),
    ((5,), 2),
    ((5,), -2),
]


@pytest.mark.parametrize("shape, axis", unsqueeze_error_params)
def test_unsqueeze_errors(shape, axis):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)

    with pytest.raises(RuntimeError):
        jetdl.unsqueeze(j_tensor, axis=axis)
