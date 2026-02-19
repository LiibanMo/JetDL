"""Tests for transcendental, trigonometric, hyperbolic, and comparison ops.

Each test runs forward values and (where applicable) backward gradients
against PyTorch as the reference implementation.
"""

import pytest
import torch

import jetdl

from ..utils import SEED, PyTestAsserts, generate_random_data, generate_shape_ids

torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------

shapes = [
    (5,),
    (10,),
    (3, 4),
    (2, 3, 4),
    (2, 3, 4, 5),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positive_data(shape):
    """Return data strictly > 0 (needed for log functions)."""
    return (torch.rand(shape) + 0.1).tolist()


def _make_mixed_data(shape):
    """Return data in [-1, 1] range (useful for trig/hyperbolic)."""
    return (torch.rand(shape) * 2 - 1).tolist()


# ---------------------------------------------------------------------------
# Forward-only registry (op_name -> (jetdl_fn, torch_fn, data_fn))
# ---------------------------------------------------------------------------

forward_ops = {
    "exp":   (jetdl.exp,   torch.exp,   generate_random_data),
    "log":   (jetdl.log,   torch.log,   _make_positive_data),
    "log10": (jetdl.log10, torch.log10, _make_positive_data),
    "log2":  (jetdl.log2,  torch.log2,  _make_positive_data),
    "sin":   (jetdl.sin,   torch.sin,   _make_mixed_data),
    "cos":   (jetdl.cos,   torch.cos,   _make_mixed_data),
    "tanh":  (jetdl.tanh,  torch.tanh,  _make_mixed_data),
    "sinh":  (jetdl.sinh,  torch.sinh,  _make_mixed_data),
    "cosh":  (jetdl.cosh,  torch.cosh,  _make_mixed_data),
    "abs":   (jetdl.abs,   torch.abs,   _make_mixed_data),
    "sign":  (jetdl.sign,  torch.sign,  _make_mixed_data),
}

forward_op_names = list(forward_ops.keys())


@pytest.mark.parametrize("op", forward_op_names)
@pytest.mark.parametrize("shape", shapes, ids=generate_shape_ids)
def test_forward(op, shape):
    jetdl_fn, torch_fn, data_fn = forward_ops[op]
    data = data_fn(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = jetdl_fn(j_tensor)
    t_result = torch_fn(t_tensor)

    assert_obj = PyTestAsserts(j_result, t_result)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()


# ---------------------------------------------------------------------------
# Tensor method tests (tensor.exp(), tensor.log(), etc.)
# ---------------------------------------------------------------------------

tensor_method_ops = {
    "exp":   (lambda t: t.exp(),   torch.exp,   generate_random_data),
    "log":   (lambda t: t.log(),   torch.log,   _make_positive_data),
    "log10": (lambda t: t.log10(), torch.log10, _make_positive_data),
    "log2":  (lambda t: t.log2(),  torch.log2,  _make_positive_data),
    "sin":   (lambda t: t.sin(),   torch.sin,   _make_mixed_data),
    "cos":   (lambda t: t.cos(),   torch.cos,   _make_mixed_data),
    "tanh":  (lambda t: t.tanh(),  torch.tanh,  _make_mixed_data),
    "sinh":  (lambda t: t.sinh(),  torch.sinh,  _make_mixed_data),
    "cosh":  (lambda t: t.cosh(),  torch.cosh,  _make_mixed_data),
    "abs":   (lambda t: t.abs(),   torch.abs,   _make_mixed_data),
    "sign":  (lambda t: t.sign(),  torch.sign,  _make_mixed_data),
}

tensor_method_names = list(tensor_method_ops.keys())


@pytest.mark.parametrize("op", tensor_method_names)
@pytest.mark.parametrize("shape", shapes, ids=generate_shape_ids)
def test_tensor_method(op, shape):
    jetdl_fn, torch_fn, data_fn = tensor_method_ops[op]
    data = data_fn(shape)

    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = jetdl_fn(j_tensor)
    t_result = torch_fn(t_tensor)

    assert_obj = PyTestAsserts(j_result, t_result)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()


# ---------------------------------------------------------------------------
# clamp forward + tensor method
# ---------------------------------------------------------------------------

clamp_params = [
    ((5,),       -0.5, 0.5),
    ((10,),      0.0,  1.0),
    ((3, 4),    -1.0,  1.0),
    ((2, 3, 4), -0.3,  0.3),
]


@pytest.mark.parametrize("shape, lo, hi", clamp_params)
def test_clamp_forward(shape, lo, hi):
    data = _make_mixed_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = jetdl.clamp(j_tensor, lo, hi)
    t_result = torch.clamp(t_tensor, lo, hi)

    assert_obj = PyTestAsserts(j_result, t_result)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()


@pytest.mark.parametrize("shape, lo, hi", clamp_params)
def test_clamp_tensor_method(shape, lo, hi):
    data = _make_mixed_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data, dtype=torch.float32)

    j_result = j_tensor.clamp(lo, hi)
    t_result = torch.clamp(t_tensor, lo, hi)

    assert_obj = PyTestAsserts(j_result, t_result)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()


# ---------------------------------------------------------------------------
# Backward (gradient) tests
# ---------------------------------------------------------------------------

backward_ops = {
    "exp":  (jetdl.exp,   torch.exp,   generate_random_data),
    "log":  (jetdl.log,   torch.log,   _make_positive_data),
    "sin":  (jetdl.sin,   torch.sin,   _make_mixed_data),
    "cos":  (jetdl.cos,   torch.cos,   _make_mixed_data),
    "tanh": (jetdl.tanh,  torch.tanh,  _make_mixed_data),
    "sinh": (jetdl.sinh,  torch.sinh,  _make_mixed_data),
    "cosh": (jetdl.cosh,  torch.cosh,  _make_mixed_data),
    "abs":  (jetdl.abs,   torch.abs,   _make_mixed_data),
}

backward_op_names = list(backward_ops.keys())

backward_shapes = [
    (5,),
    (3, 4),
    (2, 3, 4),
]


@pytest.mark.parametrize("op", backward_op_names)
@pytest.mark.parametrize("shape", backward_shapes, ids=generate_shape_ids)
def test_backward(op, shape):
    jetdl_fn, torch_fn, data_fn = backward_ops[op]
    data = data_fn(shape)

    # JetDL backward — use mean() to reduce to scalar (sum has no backward node)
    j_tensor = jetdl.tensor(data, requires_grad=True)
    j_result = jetdl_fn(j_tensor)
    j_result.mean().backward()

    # PyTorch backward
    t_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    t_result = torch_fn(t_tensor)
    t_result.mean().backward()

    assert_obj = PyTestAsserts(j_tensor.grad, t_tensor.grad)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()


@pytest.mark.parametrize("shape, lo, hi", clamp_params)
def test_clamp_backward(shape, lo, hi):
    data = _make_mixed_data(shape)

    j_tensor = jetdl.tensor(data, requires_grad=True)
    j_result = jetdl.clamp(j_tensor, lo, hi)
    j_result.mean().backward()

    t_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    t_result = torch.clamp(t_tensor, lo, hi)
    t_result.mean().backward()

    assert_obj = PyTestAsserts(j_tensor.grad, t_tensor.grad)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()


@pytest.mark.parametrize("shape", backward_shapes, ids=generate_shape_ids)
def test_log10_backward(shape):
    data = _make_positive_data(shape)

    j_tensor = jetdl.tensor(data, requires_grad=True)
    j_result = jetdl.log10(j_tensor)
    j_result.mean().backward()

    t_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    t_result = torch.log10(t_tensor)
    t_result.mean().backward()

    assert_obj = PyTestAsserts(j_tensor.grad, t_tensor.grad)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()


@pytest.mark.parametrize("shape", backward_shapes, ids=generate_shape_ids)
def test_log2_backward(shape):
    data = _make_positive_data(shape)

    j_tensor = jetdl.tensor(data, requires_grad=True)
    j_result = jetdl.log2(j_tensor)
    j_result.mean().backward()

    t_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    t_result = torch.log2(t_tensor)
    t_result.mean().backward()

    assert_obj = PyTestAsserts(j_tensor.grad, t_tensor.grad)
    assert assert_obj.check_basic_metadata(), assert_obj.basic_metadata_error_output()
    assert assert_obj.check_results(), assert_obj.results_error_output()
