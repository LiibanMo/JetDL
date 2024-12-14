from typing import Optional
import os
import ctypes

class C_Tensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]

class Tensor:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, 'libtensor.so')
    _C = ctypes.CDLL(lib_path)

    def __init__(self, data:Optional[list]=None):

        if data is not None:
            data, shape = self.flatten(data)

            self._c_data = (len(data) * ctypes.c_float)(*data)
            self._c_shape = (len(shape) * ctypes.c_int)(*shape)
            self._c_ndim = ctypes.c_int(len(shape))

            self.shape = shape
            self.ndim = len(shape)

            Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            Tensor._C.create_tensor.restype = ctypes.POINTER(C_Tensor)

            self.tensor = Tensor._C.create_tensor(
                self._c_data,
                self._c_shape,
                self._c_ndim,
            )

    def flatten(self, data):
        def recursively_flattening(data):
            if not isinstance(data[0], list):
                return data
            flattened_data = []
            for element in data:
                if isinstance(element[0], list):
                    flattened_data += recursively_flattening(element)
                elif isinstance(element[0], (int, float)):
                    flattened_data += element
            return flattened_data
        
        def recursively_get_shape(data:list):
            shape = []
            if isinstance(data, list):
                for sub_list in data:
                    inner_shape = recursively_get_shape(sub_list)
                shape.append(len(data))
                shape.extend(inner_shape)
            return shape
        
        flattened_data = recursively_flattening(data)
        shape = recursively_get_shape(data)

        return flattened_data, shape

    def __getitem__(self, indices:list):

        if len(indices) != self.ndim:
            raise IndexError(f"Incorrect number of indices inputted for tensor of shape {self.shape}")
        
        for i, index in enumerate(indices):
            if index >= self.shape[i]:
                raise IndexError(f"Incorrect value for index {i}. Expected index less than {self.shape[i]}. Got {index}.")
            elif index < 0:
                raise IndexError(f"Inputted an index less than 0. Unsupported.")
            
        Tensor._C.get_item.argtypes = [ctypes.POINTER(C_Tensor), ctypes.POINTER(ctypes.c_int)]
        Tensor._C.get_item.restype = ctypes.c_float
        
        indices = (len(indices) * ctypes.c_int)(*indices)
        
        item = Tensor._C.get_item(self.tensor, indices)
        return item
    
    def __add__(self, operand):
        if isinstance(operand, Tensor):
            Tensor._C.add_tensor.argtypes = [ctypes.POINTER(C_Tensor), ctypes.POINTER(C_Tensor)]
            Tensor._C.add_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.add_tensor(self.tensor, operand.tensor)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim

            return result_tensor
    
    def __sub__(self, operand):
        if isinstance(operand, Tensor):
            Tensor._C.subtract_tensor.argtypes = [ctypes.POINTER(C_Tensor), ctypes.POINTER(C_Tensor)]
            Tensor._C.subtract_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.subtract_tensor(self.tensor, operand.tensor)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim

            return result_tensor