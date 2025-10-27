from typing import Optional, Union

from ._C import Tensor
from ._C.routines import c_contiguous, c_reshape, c_squeeze, c_unsqueeze


def contiguous(input: Tensor) -> Tensor:
    """Returns a contiguous tensor containing the same data as the input.

    Args:
        input (Tensor): The input tensor.
    """
    return c_contiguous(input)


def reshape(input: Tensor, shape: Union[tuple, list]) -> Tensor:
    """Returns a tensor with the same data but a different shape.

    Args:
        input (Tensor): The tensor to be reshaped.
        shape (Union[tuple, list]): The new shape.
    """
    return c_reshape(input, shape)


def squeeze(input: Tensor, axes: Optional[Union[list, tuple]] = None) -> Tensor:
    """Removes dimensions of size 1 from the shape of a tensor.

    Args:
        input (Tensor): The input tensor.
        axes (Optional[Union[list, tuple]]): The axes to squeeze.
    """
    return c_squeeze(input, axes)


def unsqueeze(input: Tensor, axis: int) -> Tensor:
    """Adds a dimension of size 1 at the specified position.

    Args:
        input (Tensor): The input tensor.
        axis (int): The axis to unsqueeze.
    """
    return c_unsqueeze(input, axis)
