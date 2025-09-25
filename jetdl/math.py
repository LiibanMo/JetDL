from typing import Optional, Union

from ._C import Tensor
from ._C.math import (
    c_add,
    c_div,
    c_mul,
    c_sub,
    c_sum,
    c_sum_to_shape,
    c_pow,
    c_mean,
)


def add(a: Tensor, b: Tensor) -> Tensor:
    return c_add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return c_sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return c_mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    return c_div(a, b)


def pow(input: Tensor, exponent: int) -> Tensor:
    return c_pow(input, exponent)


def sum(input: Tensor, axes: Optional[Union[int, list, tuple]] = None) -> Tensor:
    return c_sum(input, axes)


def sum_to_shape(input: Tensor, shape: Union[list, tuple]) -> Tensor:
    return c_sum_to_shape(input, shape)


def mean(input: Tensor, axes: Optional[Union[int, list, tuple]] = None) -> Tensor:
    return c_mean(input, axes)
