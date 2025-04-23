from .routines import *
from .tensor import Tensor


def tensor(data=None, requires_grad: bool = True):
    return Tensor(data, requires_grad)
