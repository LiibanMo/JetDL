from typing import Optional, Union

from ._C import Tensor
from ._C.routines import c_contiguous, c_reshape, c_squeeze, c_unsqueeze


def contiguous(input: Tensor) -> Tensor:
    return c_contiguous(input)


def reshape(input: Tensor, shape: Union[tuple, list]) -> Tensor:
    return c_reshape(input, shape)


def squeeze(input: Tensor, axes: Optional[Union[list, tuple]] = None) -> Tensor:
    return c_squeeze(input, axes)


def unsqueeze(input: Tensor, axis: int) -> Tensor:
    return c_unsqueeze(input, axis)
