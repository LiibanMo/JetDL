from typing import Optional, Union

from ._C import Tensor
from ._C.math import (c_add, c_div, c_mean, c_mul, c_pow, c_scalar_sqrt,
                      c_sqrt, c_sub, c_sum, c_sum_to_shape)


def add(a: Tensor, b: Tensor) -> Tensor:
    """Computes the element-wise addition of two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.
    """
    return c_add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Computes the element-wise subtraction of two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.
    """
    return c_sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Computes the element-wise multiplication of two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.
    """
    return c_mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    """Computes the element-wise division of two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.
    """
    return c_div(a, b)


def pow(input: Tensor, exponent: int) -> Tensor:
    """Computes the element-wise power of a tensor.

    Args:
        input (Tensor): The input tensor.
        exponent (int): The exponent.
    """
    return c_pow(input, exponent)


def sqrt(input: Union[int, float, Tensor]) -> Union[int, float, Tensor]:
    """Computes the element-wise square root of a tensor or a scalar.

    Args:
        input (Union[int, float, Tensor]): The input tensor or scalar.
    """
    if isinstance(input, (int, float)):
        return c_scalar_sqrt(input)
    else:
        return c_sqrt(input)


def sum(input: Tensor, axes: Optional[Union[int, list, tuple]] = None) -> Tensor:
    """Computes the sum of tensor elements over given axes.

    Args:
        input (Tensor): The input tensor.
        axes (Optional[Union[int, list, tuple]]): The axes to reduce.
    """
    return c_sum(input, axes)


def sum_to_shape(input: Tensor, shape: Union[list, tuple]) -> Tensor:
    """Sums the tensor to a desired shape.

    Args:
        input (Tensor): The input tensor.
        shape (Union[list, tuple]): The desired shape.
    """
    return c_sum_to_shape(input, shape)


def mean(input: Tensor, axes: Optional[Union[int, list, tuple]] = None) -> Tensor:
    """Computes the mean of tensor elements over given axes.

    Args:
        input (Tensor): The input tensor.
        axes (Optional[Union[int, list, tuple]]): The axes to reduce.
    """
    return c_mean(input, axes)


__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "sum",
    "sum_to_shape",
    "mean",
]
