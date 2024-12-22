import torch

from python import Tensor


def compare_tensors(tensor1: torch.tensor, tensor2: torch.tensor, epsilon=1e-5) -> bool:
    diff = torch.abs(tensor1 - tensor2)
    return torch.all(diff < epsilon)
