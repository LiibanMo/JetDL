from ._C import Tensor


def tensor(data, *, requires_grad=False) -> Tensor:
    return Tensor(data, requires_grad)
