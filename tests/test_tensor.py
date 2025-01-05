import pytest
import torch

from tensorlite import Tensor

from .utils import compare_tensors, matmul_data, matmul_ids


# ---------------------------------------------------------------------------------------------------------

# Test getting item

@pytest.mark.parametrize(
    "data, idx",
    [
        ([1.0, 2.0, 3.0], 0),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (0, 1)),
        ([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], (0, 1, 2)),
    ],
)
def test_retrieving_item(data, idx):
    tensor = Tensor(data)
    result_value = tensor[idx]

    torch_tensor = torch.tensor(data)
    expected_value = torch_tensor[idx]

    assert abs(result_value - expected_value) < 1e-5


# Testing Addition


@pytest.mark.parametrize(
    "data, scalar",
    [([1, 2], 1.0), ([[1.0, 2.0], [3.0, 4.0]], 1.0), ([[[1.0, 2.0], [3.0, 4.0]]], 1.0)],
)
def test_tensor_add_scalar(data: list, scalar: float):
    tensor = Tensor(data)
    result_tensor = tensor + scalar
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor = torch.tensor(data)
    torch_expected_tensor = torch_tensor + scalar

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


@pytest.mark.parametrize(
    "data1, data2",
    [
        ([1.0, 2.0], [3.0, 4.0]),
        (
            [
                [
                    1.0,
                    2.0,
                ],
                [3.0, 4.0],
            ],
            [[1.0, 1.0], [1.0, 1.0]],
        ),
        ([[[1.0, 2.0], [3.0, 4.0]]], [[[1.0, 1.0], [1.0, 1.0]]]),
    ],
)
def test_tensor_add_tensor(data1: list, data2: list):
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 + tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 + torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


# Testing Subtraction


@pytest.mark.parametrize(
    "data, scalar",
    [
        ([1.0, 2.0], 1.0),
        ([[1.0, 2.0], [3.0, 4.0]], 1.0),
        ([[[1.0, 2.0], [3.0, 4.0]]], 1.0),
    ],
)
def test_tensor_sub_scalar(data: list, scalar: float):
    tensor = Tensor(data)
    result_tensor = tensor - scalar
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor = torch.tensor(data)
    torch_expected_tensor = torch_tensor - scalar

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


@pytest.mark.parametrize(
    "data1, data2",
    [
        ([1.0, 2.0], [3.0, 4.0]),
        (
            [
                [
                    1.0,
                    2.0,
                ],
                [3.0, 4.0],
            ],
            [[1.0, 1.0], [1.0, 1.0]],
        ),
        ([[[1.0, 2.0], [3.0, 4.0]]], [[[1.0, 1.0], [1.0, 1.0]]]),
    ],
)
def test_tensor_sub_tensor(data1: list, data2: list):
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 - tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 - torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


# Testing element-wise muliplication


@pytest.mark.parametrize(
    "data, scalar",
    [
        ([1.0, 2.0], 2.0),
        ([[1.0, 2.0], [3.0, 4.0]], 2.0),
        ([[[1.0, 2.0], [3.0, 4.0]]], 2.0),
    ],
)
def test_tensor_mul_scalar(data: list, scalar: float):
    tensor = Tensor(data)
    result_tensor = tensor * scalar
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor = torch.tensor(data)
    torch_expected_tensor = torch_tensor * scalar

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


@pytest.mark.parametrize(
    "data1, data2",
    [
        ([1.0, 2.0], [3.0, 4.0]),
        ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
        ([[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]),
    ],
)
def test_tensor_mul_tensor(data1: list, data2: list):
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 * tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 * torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


# Testing matmul ------------------------------------------------------------------------------


@pytest.mark.parametrize("data1, data2", matmul_data, ids = matmul_ids)
def test_matmul_tensor(data1: list, data2: list) -> None:
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 @ tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 @ torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


# Testing division -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, divisor",
    [
        ([1.0, 2.0, 3.0], 4.0),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 7.0),
        ([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], 7.0),
    ],
)
def test_tensor_div_scalar(data: list, divisor: float):
    tensor = Tensor(data)
    result_tensor = tensor / divisor
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor = torch.tensor(data)
    torch_expected_result = torch_tensor / divisor

    assert compare_tensors(torch_result_tensor, torch_expected_result)
