from typing import Union

numeric = Union[int, float]

class Tensor:
    shape: list[int]
    ndim: int
    strides: list[int]
    data: list[float]
    size: int
    requires_grad: bool

    def __init__(self, data: list[numeric], requires_grad: bool) -> None: ...

def tensor(data: list[numeric], requires_grad: bool) -> Tensor: ...