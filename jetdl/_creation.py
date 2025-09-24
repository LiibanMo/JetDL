from typing import Union

from ._C import Tensor
from ._C.routines import (
    c_ones,
    c_zeros,
)


def zeros(shape: Union[tuple, list], *, requires_grad: bool = False) -> Tensor:
    return c_zeros(shape, requires_grad)


def ones(shape: Union[tuple, list], *, requires_grad: bool = False) -> Tensor:
    return c_ones(shape, requires_grad)
