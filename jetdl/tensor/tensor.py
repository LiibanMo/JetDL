from typing import Union

from .._C import C_Tensor, c_init_tensor, c_destroy_tensor

class Tensor(C_Tensor):
    def __new__(cls, data):
        return c_init_tensor(data)
    
    def __del__(self):
        c_destroy_tensor(self)