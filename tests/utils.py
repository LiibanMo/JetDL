import torch


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, epsilon=1e-6) -> bool:
    diff = torch.abs(tensor1 - tensor2)
    return torch.all(diff < epsilon)
