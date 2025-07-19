from .._Cpp import c_add
from ..tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Adds two tensors element-wise.
    This function performs element-wise addition of two tensors, with support for broadcasting.

    Parameters
    ----------
    a : Tensor
        First input tensor
    b : Tensor 
        Second input tensor to be added to first tensor

    Returns
    -------
    Tensor
        A new tensor containing the element-wise sum of a and b.
        The output tensor has the same shape as the broadcasting of inputs.

    Examples
    --------
    >>> a = Tensor([[1, 2], [3, 4]])
    >>> b = Tensor([[5, 6], [7, 8]])
    >>> c = add(a, b)
    >>> print(c)
    tensor([[6, 8],
            [10, 12]])
            
    Notes
    -----
    - Input shapes must be broadcastable to each other
    - Output dtype is determined by input promotion rules
    """
    return c_add(a, b)
