from .._Cpp import c_add, c_sub, c_mul, c_div
from ..tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Adds two tensors element-wise.

    Args:
    ----
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
    --------
        Tensor: A new tensor with the result of the element-wise addition.

    Examples:
    --------
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> add(a, b)
        Tensor([5, 7, 9])
    """

    return c_add(a, b)

def sub(a: Tensor, b: Tensor) -> Tensor:
    """
    Subtracts two tensors element-wise.

    Args:
    ----
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
    --------
        Tensor: A new tensor with the result of the element-wise subtraction.

    Examples:
    --------
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> sub(a, b)
        Tensor([-3, -3, -3])
    """

    return c_sub(a, b)

def mul(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiplies two tensors element-wise.

    Args:
    ----
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
    --------
        Tensor: A new tensor with the result of the element-wise multiplication.

    Examples:
    --------
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> mul(a, b)
        Tensor([4, 10, 18])
    """

    return c_mul(a, b)

def div(a: Tensor, b: Tensor) -> Tensor:
    """
    Divides two tensors element-wise.

    Args:
    ----
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
    --------
        Tensor: A new tensor with the result of the element-wise division.

    Examples:
    --------
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> div(a, b)
        Tensor([4, 10, 18])
    """

    return c_div(a, b)