from typing import Union

from .._C import (
    TensorBase, c_destroy_tensor,
)

class Tensor(TensorBase):   
    def __del__(self):
        c_destroy_tensor(self)

    def __matmul__(self: "Tensor", other: "Tensor") -> "Tensor":
        from ..linalg import dot
        return dot(self, other)