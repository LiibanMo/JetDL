import pytest

from jetdl import Tensor


def test_tensor_creation():
    tensor = Tensor([1, 2, 3])
    assert tensor._data == [1, 2, 3]
    assert tensor.shape == (3,)
    assert tensor.ndim == 1


def test_tensor_addition():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 + tensor2
    assert result._data == [5, 7, 9]
    assert result.shape == (3,)
    assert result.ndim == 1


def test_tensor_scalar_addition():
    tensor = Tensor([1, 2, 3])
    result = tensor + 5
    assert result._data == [6, 7, 8]
    assert result.shape == (3,)
    assert result.ndim == 1


def test_tensor_broadcasting_addition():
    tensor1 = Tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = Tensor([1, 2, 3])
    result = tensor1 + tensor2
    assert result._data == [2, 4, 6, 5, 7, 9]
    assert result.shape == (2, 3)
    assert result.ndim == 2


def test_tensor_subtraction():
    tensor1 = Tensor([5, 7, 9])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 - tensor2
    assert result._data == [1, 2, 3]
    assert result.shape == (3,)
    assert result.ndim == 1


def test_tensor_scalar_subtraction():
    tensor = Tensor([5, 7, 9])
    result = tensor - 5
    assert result._data == [0, 2, 4]
    assert result.shape == (3,)
    assert result.ndim == 1


def test_tensor_broadcasting_subtraction():
    tensor1 = Tensor([[5, 7, 9], [10, 12, 14]])
    tensor2 = Tensor([1, 2, 3])
    result = tensor1 - tensor2
    assert result._data == [4, 5, 6, 9, 10, 11]
    assert result.shape == (2, 3)
    assert result.ndim == 2


def test_tensor_multiplication():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 * tensor2
    assert result._data == [4, 10, 18]
    assert result.shape == (3,)
    assert result.ndim == 1


def test_tensor_scalar_multiplication():
    tensor = Tensor([1, 2, 3])
    result = tensor * 5
    assert result._data == [5, 10, 15]
    assert result.shape == (3,)
    assert result.ndim == 1


def test_tensor_broadcasting_multiplication():
    tensor1 = Tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = Tensor([2, 3, 4])
    result = tensor1 * tensor2
    assert result._data == [2, 6, 12, 8, 15, 24]
    assert result.shape == (2, 3)
    assert result.ndim == 2


def test_tensor_matmul():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[5, 6], [7, 8]])
    result = tensor1 @ tensor2
    assert result._data == [19, 22, 43, 50]
    assert result.shape == (2, 2)
    assert result.ndim == 2


def test_tensor_broadcasting_matmul():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([5, 6])
    result = tensor1 @ tensor2
    assert result._data == [17, 39]
    assert result.shape == (2,)
    assert result.ndim == 1


def test_tensor_3d_matmul_with_matrix():
    tensor1 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor2 = Tensor([[1, 2], [3, 4]])
    result = tensor1 @ tensor2
    assert result._data == [7, 10, 15, 22, 23, 34, 31, 46]
    assert result.shape == (2, 2, 2)
    assert result.ndim == 3


def test_tensor_reshape():
    tensor = Tensor([1, 2, 3, 4])
    reshaped = tensor.reshape([2, 2])
    assert reshaped._data == [1, 2, 3, 4]
    assert reshaped.shape == (2, 2)
    assert reshaped.ndim == 2


def test_tensor_flatten():
    tensor = Tensor([[1, 2], [3, 4]])
    flattened = tensor.flatten()
    assert flattened._data == [1, 2, 3, 4]
    assert flattened.shape == (4,)
    assert flattened.ndim == 1


def test_tensor_sum():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor.sum()
    assert result._data == [10]
    assert result.shape == ()
    assert result.ndim == 0


def test_tensor_broadcasting_sum():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    result = tensor.sum(axis=0)
    assert result._data == [5, 7, 9]
    assert result.shape == (3,)
    assert result.ndim == 1

    result = tensor.sum(axis=1)
    assert result._data == [6, 15]
    assert result.shape == (2,)
    assert result.ndim == 1


def test_tensor_mean():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor.mean()
    assert pytest.approx(result._data[0]) == 2.5
    assert result.shape == ()
    assert result.ndim == 0


def test_tensor_broadcasting_mean():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    result = tensor.mean(axis=0)
    assert pytest.approx(result._data) == [2.5, 3.5, 4.5]
    assert result.shape == (3,)
    assert result.ndim == 1

    result = tensor.mean(axis=1)
    assert pytest.approx(result._data) == [2.0, 5.0]
    assert result.shape == (2,)
    assert result.ndim == 1


def test_tensor_power():
    tensor = Tensor([1, 2, 3])
    result = tensor**2
    assert result._data == [1, 4, 9]
    assert result.shape == (3,)
    assert result.ndim == 1


def test_tensor_backward():
    tensor = Tensor([1, 2, 3], requires_grad=True)
    result = tensor * 2
    result.backward()
    assert tensor.grad._data == [2, 2, 2]
