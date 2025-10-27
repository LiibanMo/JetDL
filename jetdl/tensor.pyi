from __future__ import annotations
from typing import Union, TypeAlias, Sequence, overload

Numeric: TypeAlias = Union[int, float]
Axes: TypeAlias = Union[int, Sequence[int]]

class Tensor:
    """
    A multi-dimensional matrix containing elements of a single data type.
    """

    # Initialization
    @overload
    def __init__(self, data: Union[Numeric, Sequence], *, requires_grad: bool = False) -> None: ...
    @overload
    def __init__(self, data: Tensor, *, requires_grad: bool = False) -> None: ...

    # Properties
    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor."""
        ...

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        ...

    @property
    def strides(self) -> tuple[int, ...]:
        """Returns the strides of the tensor."""
        ...

    @property
    def is_contiguous(self) -> bool:
        """Returns whether the tensor is contiguous in memory."""
        ...

    @property
    def requires_grad(self) -> bool:
        """Returns whether the tensor requires gradients to be computed."""
        ...

    @property
    def grad(self) -> Tensor | None:
        """Returns the gradient of the tensor. Is `None` if `requires_grad` is `False`."""
        ...

    @property
    def T(self) -> Tensor:
        """Returns a transposed version of this tensor."""
        ...

    @property
    def mT(self) -> Tensor:
        """Returns a matrix-transposed version of this tensor."""
        ...

    # Dunder methods
    def __str__(self) -> str: ...
    def __neg__(self) -> Tensor: ...

    # --- Arithmetic ---
    def __add__(self, other: Tensor | Numeric) -> Tensor: ...
    def __radd__(self, other: Numeric) -> Tensor: ...

    def __sub__(self, other: Tensor | Numeric) -> Tensor: ...
    def __rsub__(self, other: Numeric) -> Tensor: ...

    def __mul__(self, other: Tensor | Numeric) -> Tensor: ...
    def __rmul__(self, other: Numeric) -> Tensor: ...

    def __truediv__(self, other: Tensor | Numeric) -> Tensor: ...
    def __rtruediv__(self, other: Numeric) -> Tensor: ...

    def __pow__(self, exponent: Numeric) -> Tensor: ...

    # --- Linalg ---
    def __matmul__(self, other: Tensor) -> Tensor: ...

    # --- Autograd ---
    def backward(self) -> None:
        """
        Computes the gradient of this tensor with respect to graph leaves.
        The graph is differentiated using the chain rule.
        """
        ...

    # --- Math ---
    def sum(self, axes: Axes | None = None) -> Tensor:
        """
        Returns the sum of each element in the input tensor.
        Can be performed along specified axes.
        """
        ...

    def sum_to_shape(self, shape: tuple[int, ...]) -> Tensor:
        """
        Sums the tensor to a desired shape.
        """
        ...

    def mean(self, axes: Axes | None = None) -> Tensor:
        """
        Returns the mean of each element in the input tensor.
        Can be performed along specified axes.
        """
        ...

    # --- Routines ---
    def reshape(self, new_shape: tuple[int, ...]) -> Tensor:
        """
        Returns a tensor with the same data and number of elements as self but with the specified shape.
        """
        ...

    def squeeze(self, axes: Axes | None = None) -> Tensor:
        """
        Returns a tensor with all the dimensions of input of size 1 removed.
        """
        ...

    def unsqueeze(self, axis: int) -> Tensor:
        """
        Returns a new tensor with a dimension of size one inserted at the specified position.
        """
        ...
