from typing import Optional, Union
from ..tensor import Tensor

def sum(input: Tensor, axes: Optional[Union[list[int], tuple[int], int]] = None) -> Tensor: ...