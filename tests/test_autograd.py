import pytest
from tensorlite.tensor import Tensor
from tensorlite.autograd import AddBackward, SubBackward, MulBackward, MmBackward, PowBackward, MeanBackward

def test_add_backward():
    tensorA = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    tensorB = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    result_tensor = Tensor([5.0, 7.0, 9.0], requires_grad=True)
    result_tensor.grad = Tensor([1.0, 1.0, 1.0])

    backward = AddBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA.data == pytest.approx([1.0, 1.0, 1.0])
    assert gradB.data == pytest.approx([1.0, 1.0, 1.0])

def test_sub_backward():
    tensorA = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    tensorB = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    result_tensor = Tensor([-3.0, -3.0, -3.0], requires_grad=True)
    result_tensor.grad = Tensor([1.0, 1.0, 1.0])

    backward = SubBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA.data == pytest.approx([1.0, 1.0, 1.0])
    assert gradB.data == pytest.approx([-1.0, -1.0, -1.0])

def test_mul_backward():
    tensorA = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    tensorB = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    result_tensor = Tensor([4.0, 10.0, 18.0], requires_grad=True)
    result_tensor.grad = Tensor([1.0, 1.0, 1.0])

    backward = MulBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA.data == pytest.approx([4.0, 5.0, 6.0])
    assert gradB.data == pytest.approx([1.0, 2.0, 3.0])

def test_mm_backward():
    tensorA = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    tensorB = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    result_tensor = Tensor([[19.0, 22.0], [43.0, 50.0]], requires_grad=True)
    result_tensor.grad = Tensor([[1.0, 1.0], [1.0, 1.0]])

    backward = MmBackward(tensorA, tensorB, result_tensor)
    gradA, gradB = backward.backward()

    expected_gradA = Tensor([[11.0, 15.0], [11.0, 15.0]])
    expected_gradB = Tensor([[4.0, 4.0], [6.0, 6.0]])

    assert gradA.shape == expected_gradA.shape
    assert gradA.flatten().data == expected_gradA.flatten().data
    assert gradB.shape == expected_gradB.shape
    assert gradB.flatten().data == expected_gradB.flatten().data

def test_pow_backward():
    tensorA = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    exponent = 3
    result_tensor = Tensor([8.0, 27.0, 64.0], requires_grad=True)
    result_tensor.grad = Tensor([1.0, 1.0, 1.0])

    backward = PowBackward(tensorA, exponent, result_tensor)
    gradA, gradB = backward.backward()

    assert gradA.data == pytest.approx([12.0, 27.0, 48.0])
    assert gradB is None

def test_mean_backward():
    tensorA = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    result_tensor = Tensor(2.0, requires_grad=True)
    result_tensor.grad = Tensor(1.0)

    backward = MeanBackward(tensorA, axis=None, result_tensor=result_tensor)
    gradA, gradB = backward.backward()

    assert gradA.data == pytest.approx([1/3, 2/3, 1])
    assert gradB is None