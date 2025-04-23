import pytest

from tensorlite.tensor import Tensor


def test_tensor_creation():
    tensor = Tensor([1, 2, 3])
    assert tensor.data == [1, 2, 3]
    assert tensor.shape == (3,)
    assert tensor.ndim == 1


def test_tensor_addition():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 + tensor2
    assert result.data == [5, 7, 9]
    assert result.shape == (3,)


def test_tensor_scalar_addition():
    tensor = Tensor([1, 2, 3])
    result = tensor + 5
    assert result.data == [6, 7, 8]
    assert result.shape == (3,)


def test_tensor_subtraction():
    tensor1 = Tensor([5, 7, 9])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 - tensor2
    assert result.data == [1, 2, 3]
    assert result.shape == (3,)


def test_tensor_scalar_subtraction():
    tensor = Tensor([5, 7, 9])
    result = tensor - 5
    assert result.data == [0, 2, 4]
    assert result.shape == (3,)


def test_tensor_multiplication():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 * tensor2
    assert result.data == [4, 10, 18]
    assert result.shape == (3,)


def test_tensor_scalar_multiplication():
    tensor = Tensor([1, 2, 3])
    result = tensor * 3
    assert result.data == [3, 6, 9]
    assert result.shape == (3,)


def test_tensor_matmul():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[5, 6], [7, 8]])
    result = tensor1 @ tensor2
    assert result.data == [19, 22, 43, 50]
    assert result.shape == (2, 2)


def test_tensor_reshape():
    tensor = Tensor([1, 2, 3, 4])
    reshaped = tensor.reshape([2, 2])
    assert reshaped.data == [1, 2, 3, 4]
    assert reshaped.shape == (2, 2)


def test_tensor_flatten():
    tensor = Tensor([[1, 2], [3, 4]])
    flattened = tensor.flatten()
    assert flattened.data == [1, 2, 3, 4]
    assert flattened.shape == (4,)


def test_tensor_sum():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor.sum()
    assert result.data == [10]
    assert result.shape == ()


def test_tensor_mean():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor.mean()
    assert result.data == [2.5]
    assert result.shape == ()


def test_tensor_power():
    tensor = Tensor([1, 2, 3])
    result = tensor**2
    assert result.data == [1, 4, 9]
    assert result.shape == (3,)


def test_tensor_backward():
    tensor = Tensor([1, 2, 3], requires_grad=True)
    result = tensor * 2
    result.backward()
    assert tensor.grad.data == [2, 2, 2]
