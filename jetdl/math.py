from typing import Optional, Union

from ._C import Tensor
from ._C.math import (c_abs, c_add, c_clamp, c_cos, c_cosh, c_div, c_exp,
                      c_log, c_log10, c_log2, c_mean, c_mul, c_pow,
                      c_scalar_sqrt, c_sign, c_sin, c_sinh, c_sqrt, c_sub,
                      c_sum, c_sum_to_shape, c_tanh)


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


def exp(input: Tensor) -> Tensor:
    """Computes the element-wise exponential of a tensor."""
    return c_exp(input)


def log(input: Tensor) -> Tensor:
    """Computes the element-wise natural logarithm of a tensor."""
    return c_log(input)


def log10(input: Tensor) -> Tensor:
    """Computes the element-wise base-10 logarithm of a tensor."""
    return c_log10(input)


def log2(input: Tensor) -> Tensor:
    """Computes the element-wise base-2 logarithm of a tensor."""
    return c_log2(input)


def sin(input: Tensor) -> Tensor:
    """Computes the element-wise sine of a tensor."""
    return c_sin(input)


def cos(input: Tensor) -> Tensor:
    """Computes the element-wise cosine of a tensor."""
    return c_cos(input)


def tanh(input: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic tangent of a tensor."""
    return c_tanh(input)


def sinh(input: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic sine of a tensor."""
    return c_sinh(input)


def cosh(input: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic cosine of a tensor."""
    return c_cosh(input)


def abs(input: Tensor) -> Tensor:
    """Computes the element-wise absolute value of a tensor."""
    return c_abs(input)


def sign(input: Tensor) -> Tensor:
    """Computes the element-wise sign of a tensor (-1, 0, or 1)."""
    return c_sign(input)


def clamp(input: Tensor, min: float, max: float) -> Tensor:
    """Clamps all elements in a tensor into the range [min, max]."""
    return c_clamp(input, min, max)


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
    "exp",
    "log",
    "log10",
    "log2",
    "sin",
    "cos",
    "tanh",
    "sinh",
    "cosh",
    "abs",
    "sign",
    "clamp",
]
