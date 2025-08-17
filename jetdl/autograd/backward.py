from typing import Optional

from .._Cpp import c_backward
from ..tensor import Tensor

def backward(input_tensor: "Tensor"):
    input_grad = Tensor([1]) #
    return c_backward(input_tensor, input_grad)