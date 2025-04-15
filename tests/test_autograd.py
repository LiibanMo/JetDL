import pytest
import torch

from tensorlite import tensor

from .test_autograd_data import matmul_backward_data, matmul_backward_ids
from .utils import compare_tensors


@pytest.mark.parametrize("data1, data2", matmul_backward_data, ids=matmul_backward_ids)
def test_matmul_backward(data1, data2):
    tensor1 = tensor(data1)
    tensor2 = tensor(data2)
    result_tensor = tensor1 @ tensor2
    result_tensor.backward()

    expected_tensor1_grad = torch.tensor(tensor1.grad.data).reshape(tensor1.grad.shape)
    expected_tensor2_grad = torch.tensor(tensor2.grad.data).reshape(tensor2.grad.shape)

    torch_tensor1 = torch.tensor(data1, dtype=torch.float32, requires_grad=True)
    torch_tensor2 = torch.tensor(data2, dtype=torch.float32, requires_grad=True)
    result_torch_tensor = torch_tensor1 @ torch_tensor2
    result_torch_tensor.backward(torch.ones_like(result_torch_tensor))

    compare_tensors(expected_tensor1_grad, torch_tensor1.grad)
    compare_tensors(expected_tensor2_grad, torch_tensor2.grad)
