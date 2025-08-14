from typing import Optional, Union

from .._Cpp import c_sum_w_axes, c_total_sum
from ..tensor import Tensor

def sum(input: Tensor, axes: Optional[Union[list[int], tuple[int], int]] = None) -> Tensor:
    """
    Calculates the sum of all elements in the input tensor, or along specified axes.

    This function behaves similarly to `torch.sum`.

    Args:
        input (Tensor): The input tensor.
        axes (Optional[Union[list[int], tuple[int], int]]): The axis or axes along which to sum.
            If None, the sum of all elements in the input tensor is returned.
            If an int, sums along the specified axis.
            If a list or tuple of ints, sums along all specified axes.

    Returns:
        Tensor: The resulting tensor after summation.
    """
    if axes is None:
        return c_total_sum(input)
    
    if isinstance(axes, int):
        input_axes = [axes]
    elif isinstance(axes, tuple):
        input_axes = list(axes)
    else:
        input_axes = axes
    return c_sum_w_axes(input, input_axes)