from typing import Union

from ._C import Tensor
from ._C.routines import c_fill, c_ones, c_zeros


def zeros(
    shape: Union[tuple, list],
    *,
    requires_grad: bool = False,
    device: str = "cpu"
) -> Tensor:
    """Creates a tensor of a given shape filled with zeros.

    Args:
        shape (Union[tuple, list]): The shape of the output tensor.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        device (str): The device to create the tensor on ('cpu' or 'cuda').
    """
    return c_zeros(shape, requires_grad, device)


def ones(
    shape: Union[tuple, list],
    *,
    requires_grad: bool = False,
    device: str = "cpu"
) -> Tensor:
    """Creates a tensor of a given shape filled with ones.

    Args:
        shape (Union[tuple, list]): The shape of the output tensor.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        device (str): The device to create the tensor on ('cpu' or 'cuda').
    """
    return c_ones(shape, requires_grad, device)


def fill(
    shape: Union[tuple, list],
    value: Union[int, float],
    *,
    requires_grad: bool = False,
    device: str = "cpu"
) -> Tensor:
    """Creates a tensor of a given shape filled with a specified scalar value.

    Args:
        shape (Union[tuple, list]): The shape of the output tensor.
        value (Union[int, float]): The value to fill the tensor with.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        device (str): The device to create the tensor on ('cpu' or 'cuda').
    """
    return c_fill(shape, value, requires_grad, device)
