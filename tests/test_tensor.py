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

def test_tensor_transpose():
    tensor = Tensor([[[1, 2, 3], [4, 5, 6]]])
    transposed = tensor.T
    assert transposed.data == [1, 4, 2, 5, 3, 6]
    assert transposed.shape == (3, 2, 1)


def test_tensor_matrix_transpose():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    transposed = tensor.mT
    assert transposed.data == [1, 4, 2, 5, 3, 6]
    assert transposed.shape == (3, 2)


def test_tensor_sum_axis():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor.sum(axis=0)
    assert result.data == [4, 6]
    assert result.shape == (2,)


def test_tensor_mean_axis():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor.mean(axis=1)
    assert result.data == pytest.approx([1.5, 3.5])
    assert result.shape == (2,)


def test_tensor_copy():
    tensor = Tensor([1, 2, 3])
    copied_tensor = tensor.copy()
    assert copied_tensor.data == tensor.data
    assert copied_tensor.shape == tensor.shape
    assert copied_tensor is not tensor


def test_tensor_broadcasting_addition():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([1, 2])
    result = tensor1 + tensor2
    assert result.data == [2, 4, 4, 6]
    assert result.shape == (2, 2)


def test_tensor_broadcasting_subtraction():
    tensor1 = Tensor([[5, 6], [7, 8]])
    tensor2 = Tensor([1, 2])
    result = tensor1 - tensor2
    assert result.data == [4, 4, 6, 6]
    assert result.shape == (2, 2)


def test_tensor_broadcasting_multiplication():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([2, 3])
    result = tensor1 * tensor2
    assert result.data == [2, 6, 6, 12]
    assert result.shape == (2, 2)


def test_tensor_division():
    tensor = Tensor([4, 9, 16])
    result = tensor / 2
    assert result.data == pytest.approx([2, 4.5, 8])
    assert result.shape == (3,)


def test_tensor_broadcasting_division():
    tensor1 = Tensor([[4, 9], [16, 25]])
    tensor2 = Tensor([2, 3])
    result = tensor1 / tensor2
    assert result.data == pytest.approx([2, 3, 8, 8.333333333333334])
    assert result.shape == (2, 2)
