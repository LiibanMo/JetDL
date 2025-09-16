from typing import Union

from ._C import Tensor
from ._C.routines import c_ones


def ones(shape: Union[tuple, list], *, requires_grad: bool = True) -> Tensor:
    return c_ones(shape, requires_grad)
