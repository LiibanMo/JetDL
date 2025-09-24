import pytest
import torch

import jetdl
from ..utils import (
    SEED,
    PyTestAsserts,
    generate_random_data,
)


torch.manual_seed(SEED)


# --- Fixtures ---


@pytest.fixture
def jetdl_tensors():
    def _get_tensors(data1, data2=None):
        if data2 is None:
            return jetdl.tensor(data1, requires_grad=True)
        return (
            jetdl.tensor(data1, requires_grad=True),
            jetdl.tensor(data2, requires_grad=True),
        )

    return _get_tensors


@pytest.fixture
def torch_tensors():
    def _get_tensors(data1, data2=None):
        if data2 is None:
            return torch.tensor(data1, requires_grad=True, dtype=torch.float32)
        return (
            torch.tensor(data1, requires_grad=True, dtype=torch.float32),
            torch.tensor(data2, requires_grad=True, dtype=torch.float32),
        )

    return _get_tensors


# --- Test Cases ---


@pytest.mark.parametrize(
    "op_str, data1, data2",
    [
        ("add", 2.0, 3.0),
        ("sub", 5.0, 2.0),
        ("mul", 4.0, 5.0),
        ("div", 10.0, 2.0),
    ],
)
def test_backward_simple_ops(op_str, data1, data2, jetdl_tensors, torch_tensors):
    j_a, j_b = jetdl_tensors(data1, data2)
    t_a, t_b = torch_tensors(data1, data2)

    # JetDL
    j_op = getattr(jetdl, op_str)
    j_c = j_op(j_a, j_b)
    j_c.backward()

    # PyTorch
    t_op = getattr(torch, op_str)
    t_c = t_op(t_a, t_b)
    t_c.backward()

    # Assertions
    assert_grad_a = PyTestAsserts(j_a.grad, t_a.grad)
    assert assert_grad_a.check_results(), assert_grad_a.results_error_output()
    assert_grad_b = PyTestAsserts(j_b.grad, t_b.grad)
    assert assert_grad_b.check_results(), assert_grad_b.results_error_output()


def test_backward_dot(jetdl_tensors, torch_tensors):
    data1, data2 = generate_random_data((3,), (3,))
    j_a, j_b = jetdl_tensors(data1, data2)
    t_a, t_b = torch_tensors(data1, data2)

    # JetDL
    j_c = jetdl.dot(j_a, j_b)
    j_c.backward()

    # PyTorch
    t_c = torch.dot(t_a, t_b)
    t_c.backward()

    # Assertions
    assert_grad_a = PyTestAsserts(j_a.grad, t_a.grad)
    assert assert_grad_a.check_results(), assert_grad_a.results_error_output()
    assert_grad_b = PyTestAsserts(j_b.grad, t_b.grad)
    assert assert_grad_b.check_results(), assert_grad_b.results_error_output()


def test_backward_matmul_chain(jetdl_tensors, torch_tensors):
    data1, data2 = generate_random_data((2, 3), (3, 4))
    j_a = jetdl_tensors(data1)
    j_b = jetdl_tensors(data2)
    t_a = torch_tensors(data1)
    t_b = torch_tensors(data2)

    data3, data4 = generate_random_data(2, 4)
    j_c, j_d = jetdl_tensors(data3, data4)
    t_c, t_d = torch_tensors(data3, data4)

    # JetDL
    j1 = j_a @ j_b
    j2 = j_c @ j1
    j3 = j2 @ j_d
    j3.backward()

    # PyTorch
    t1 = t_a @ t_b
    t2 = t_c @ t1
    t3 = t2 @ t_d
    t3.backward()

    # Assertions
    assert_grad_a = PyTestAsserts(j_a.grad, t_a.grad)
    assert assert_grad_a.check_results(), assert_grad_a.results_error_output()
    assert_grad_b = PyTestAsserts(j_b.grad, t_b.grad)
    assert assert_grad_b.check_results(), assert_grad_b.results_error_output()


def test_backward_arith_chain(jetdl_tensors, torch_tensors):
    j_a, j_b = jetdl_tensors(2.0, 3.0)
    j_d, j_e = jetdl_tensors(4.0, 5.0)

    t_a, t_b = torch_tensors(2.0, 3.0)
    t_d, t_e = torch_tensors(4.0, 5.0)

    # JetDL
    j_f = j_a * j_b
    j_g = j_d + j_f
    j_h = j_g / j_e
    j_h.backward()

    # PyTorch
    t_f = t_a * t_b
    t_g = t_d + t_f
    t_h = t_g / t_e
    t_h.backward()

    # Assertions
    for j_tensor, t_tensor in [(j_a, t_a), (j_b, t_b), (j_d, t_d), (j_e, t_e)]:
        assert_grad = PyTestAsserts(j_tensor.grad, t_tensor.grad)
        assert assert_grad.check_results(), assert_grad.results_error_output()


def test_backward_on_non_scalar():
    j_a = jetdl.tensor([1.0, 2.0, 3.0], requires_grad=True)
    with pytest.raises(RuntimeError) as err:
        j_a.backward()
    assert "backward pass only starts for scalar tensors" in str(err.value)
