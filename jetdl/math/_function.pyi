from typing import Optional
from .._tensor import Tensor

def sum(input: "Tensor", axes: Optional[list[int]] = None) -> "Tensor": ...