from typing import Optional
import pytest
import torch

from tensorlite import Tensor

from .test_utils import (
    compare_tensors, 
    without_broadcasting_data, without_broadcasting_ids,
    broadcasting_data, add_broadcasting_ids, sub_broadcasting_ids, mul_broadcasting_ids, div_broadcasting_ids,
    matmul_data, matmul_ids,
    shapes_for_T_mT, shapes_for_T_ids, shapes_for_mT_ids,
    reduction_operations_data, sum_ids,
    )

# ---------------------------------------------------------------------------------------------------------

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


@pytest.mark.parametrize("data1, data2", without_broadcasting_data, ids=without_broadcasting_ids)
def test_tensor_add_tensor(data1: list, data2: list):
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 + tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 + torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


@pytest.mark.parametrize("data1, data2", broadcasting_data, ids=add_broadcasting_ids)
def test_tensor_add_broadcasting(data1, data2) -> None:
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


@pytest.mark.parametrize("data1, data2", without_broadcasting_data, ids=without_broadcasting_ids)
def test_tensor_sub_tensor(data1: list, data2: list):
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 - tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 - torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)

@pytest.mark.parametrize("data1, data2", broadcasting_data, ids=sub_broadcasting_ids)
def test_tensor_sub_broadcasting(data1, data2) -> None:
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


@pytest.mark.parametrize("data1, data2", without_broadcasting_data, ids=without_broadcasting_ids)
def test_tensor_mul_tensor(data1: list, data2: list):
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 * tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 * torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


@pytest.mark.parametrize("data1, data2", broadcasting_data, ids=mul_broadcasting_ids)
def test_tensor_mul_broadcasting(data1, data2) -> None:
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 * tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 * torch_tensor2
   
    assert compare_tensors(torch_result_tensor, torch_expected_tensor)

# Testing matmul 

@pytest.mark.parametrize("data1, data2", matmul_data, ids=matmul_ids)
def test_matmul_tensor(data1: list, data2: list) -> None:
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 @ tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 @ torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)

# Testing division 

@pytest.mark.parametrize("data1, data2", without_broadcasting_data, ids=without_broadcasting_ids)
def test_tensor_div_tensor(data1: list, data2: list):
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 / tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 / torch_tensor2

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


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


@pytest.mark.parametrize("data1, data2", broadcasting_data, ids=div_broadcasting_ids)
def test_tensor_sub_broadcasting(data1, data2) -> None:
    tensor1 = Tensor(data1)
    tensor2 = Tensor(data2)
    result_tensor = tensor1 / tensor2
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_expected_tensor = torch_tensor1 / torch_tensor2
   
    assert compare_tensors(torch_result_tensor, torch_expected_tensor)


# Testing .T

@pytest.mark.parametrize("shape", shapes_for_T_mT, ids=shapes_for_T_ids)
def test_tensor_T(shape: list):
    data = torch.arange(24).reshape(shape).tolist()

    tensor_T = Tensor(data).T
    torch_result_tensor = torch.tensor(tensor_T.data).reshape(tensor_T.shape)

    torch_tensor = torch.tensor(data)
    T_dims = torch.arange(torch_tensor.ndim-1, -1, -1).tolist()
    torch_expected_tensor = torch_tensor.permute(T_dims)

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)

# Testing .mT

@pytest.mark.parametrize("shape", shapes_for_T_mT, ids=shapes_for_mT_ids)
def test_tensor_mT(shape:list):
    data = torch.arange(24).reshape(shape).tolist()

    tensor_mT = Tensor(data).mT
    torch_result_tensor = torch.tensor(tensor_mT.data).reshape(tensor_mT.shape)

    torch_expected_tensor = torch.tensor(data).mT

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)

# Testing sum()

@pytest.mark.parametrize("data, axis", reduction_operations_data, ids=sum_ids)
def test_tensor_sum(data:list, axis:Optional[int]):
    result_tensor = Tensor(data).sum(axis=axis)
    torch_result_tensor = torch.tensor(result_tensor.data).reshape(result_tensor.shape)

    torch_expected_tensor = torch.tensor(data).sum(axis=axis)

    assert compare_tensors(torch_result_tensor, torch_expected_tensor)