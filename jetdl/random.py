from typing import Union

from ._C.random import c_normal, c_uniform


def uniform(low: float, high: float, shape: Union[list, tuple] = [], seed: int = 123):
    """Creates a tensor with values from a uniform distribution.

    Args:
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        shape (Union[list, tuple]): The shape of the output tensor.
        seed (int): The seed for the random number generator.
    """
    return c_uniform(low, high, shape, seed)


def normal(mean: float, std: float, shape: Union[list, tuple] = [], seed: int = 123):
    """Creates a tensor with values from a normal distribution.

    Args:
        mean (float): The mean of the distribution.
        std (float): The standard deviation of the distribution.
        shape (Union[list, tuple]): The shape of the output tensor.
        seed (int): The seed for the random number generator.
    """
    return c_normal(mean, std, shape, seed)


def rand(*shape: Union[int, list, tuple], seed: int = 123):
    """Creates a tensor with values from a uniform distribution between 0 and 1.

    Args:
        shape (Union[int, list, tuple]): The shape of the output tensor.
        seed (int): The seed for the random number generator.
    """
    return c_uniform(0, 1, shape, seed)
