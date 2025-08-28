from typing import Optional, Union
from .._C.math import c_sum
from .._tensor import Tensor

def sum(input: "Tensor", axes: Optional[Union[int, float, tuple, list]] = None) -> "Tensor":
    if axes is None:
        return c_sum(input, [])
    elif isinstance(axes, (int, float)):
        return c_sum(input, [axes])
    elif isinstance(axes, tuple):
        return c_sum(input, list(axes))
    else:
        return c_sum(input, axes)