import pytest

from tensorlite.tensor import tensor
from tensorlite.autograd import (AddBackward, MeanBackward, MmBackward,
                                 MulBackward, PowBackward, SubBackward)
from tensorlite.autograd.control_utils import (
    GradMode, no_grad
)


def test_add_backward():
    tensorA = tensor([1.0, 2.0, 3.0], requires_grad=True)
    tensorB = tensor([4.0, 5.0, 6.0], requires_grad=True)
    result_tensor = tensor([5.0, 7.0, 9.0], requires_grad=True)
    result_tensor.grad = tensor([1.0, 1.0, 1.0])

    backward = AddBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA._data == pytest.approx([1.0, 1.0, 1.0])
    assert gradB._data == pytest.approx([1.0, 1.0, 1.0])


def test_sub_backward():
    tensorA = tensor([1.0, 2.0, 3.0], requires_grad=True)
    tensorB = tensor([4.0, 5.0, 6.0], requires_grad=True)
    result_tensor = tensor([-3.0, -3.0, -3.0], requires_grad=True)
    result_tensor.grad = tensor([1.0, 1.0, 1.0])

    backward = SubBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA._data == pytest.approx([1.0, 1.0, 1.0])
    assert gradB._data == pytest.approx([-1.0, -1.0, -1.0])


def test_mul_backward():
    tensorA = tensor([1.0, 2.0, 3.0], requires_grad=True)
    tensorB = tensor([4.0, 5.0, 6.0], requires_grad=True)
    result_tensor = tensor([4.0, 10.0, 18.0], requires_grad=True)
    result_tensor.grad = tensor([1.0, 1.0, 1.0])

    backward = MulBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA._data == pytest.approx([4.0, 5.0, 6.0])
    assert gradB._data == pytest.approx([1.0, 2.0, 3.0])


def test_mm_backward():
    tensorA = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    tensorB = tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    result_tensor = tensor([[19.0, 22.0], [43.0, 50.0]], requires_grad=True)
    result_tensor.grad = tensor([[1.0, 1.0], [1.0, 1.0]])

    backward = MmBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    expected_gradA = tensor([[11.0, 15.0], [11.0, 15.0]])
    expected_gradB = tensor([[4.0, 4.0], [6.0, 6.0]])

    assert gradA.shape == expected_gradA.shape
    assert gradA.flatten()._data == expected_gradA.flatten()._data
    assert gradB.shape == expected_gradB.shape
    assert gradB.flatten()._data == expected_gradB.flatten()._data


def test_pow_backward():
    tensorA = tensor([2.0, 3.0, 4.0], requires_grad=True)
    exponent = 3
    result_tensor = tensor([8.0, 27.0, 64.0], requires_grad=True)
    result_tensor.grad = tensor([1.0, 1.0, 1.0])

    backward = PowBackward(tensorA, exponent, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA._data == pytest.approx([12.0, 27.0, 48.0])
    assert gradB is None


def test_mean_backward():
    tensorA = tensor([1.0, 2.0, 3.0], requires_grad=True)
    result_tensor = tensor(2.0, requires_grad=True)
    result_tensor.grad = tensor(1.0)

    backward = MeanBackward(tensorA, axis=None, result_tensor=result_tensor)
    gradA, gradB = backward.backward()

    assert gradA._data == pytest.approx([1 / 3, 2 / 3, 1])
    assert gradB is None


def test_no_grad():
    assert GradMode.is_enabled()

    with no_grad():
        assert not GradMode.is_enabled()

    assert GradMode.is_enabled()

    with no_grad():
        with no_grad():
            assert not GradMode.is_enabled()

    assert GradMode.is_enabled()