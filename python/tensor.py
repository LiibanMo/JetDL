import ctypes
import os
from typing import Optional


class C_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
    ]


class Tensor:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "libtensor.so")
    _C = ctypes.CDLL(lib_path)

    _C.create_tensor.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    _C.create_tensor.restype = ctypes.POINTER(C_Tensor)

    _C.free_tensor.argtypes = [ctypes.POINTER(C_Tensor)]
    _C.free_tensor.restype = None

    _C.free_data.argtypes = [ctypes.POINTER(C_Tensor)]
    _C.free_data.restype = None

    _C.free_shape.argtypes = [ctypes.POINTER(C_Tensor)]
    _C.free_shape.restype = None

    _C.free_strides.argtypes = [ctypes.POINTER(C_Tensor)]
    _C.free_strides.restype = None

    def __init__(self, data: Optional[list] = None):
        if data is not None:
            data, shape = self.flatten(data)

            self._c_data = (len(data) * ctypes.c_double)(*data)
            self._c_shape = (len(shape) * ctypes.c_int)(*shape)
            self._c_ndim = ctypes.c_int(len(shape))

            self.shape = shape
            self.ndim = len(shape)
            self.data = data
            self.size = 1
            for dim in shape:
                self.size *= dim

            self.tensor = Tensor._C.create_tensor(
                self._c_data,
                self._c_shape,
                self._c_ndim,
            )

    def __del__(self):
        if hasattr(self, "tensor"):
            Tensor._C.free_tensor(self.tensor)
        elif hasattr(self, "data"):
            Tensor._C.free_data(self.data)
        elif hasattr(self, "shape"):
            Tensor._C.free_shape(self.shape)
        elif hasattr(self, "strides"):
            Tensor._C.free_strides(self.strides)

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

        def recursively_get_shape(data: list):
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

    def __getitem__(self, indices):
        Tensor._C.get_item.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(ctypes.c_int),
        ]
        Tensor._C.get_item.restype = ctypes.c_double

        if isinstance(indices, (int, float)) and self.ndim == 1:
            indices = (indices,)

        elif isinstance(indices, tuple):
            if len(indices) != self.ndim:
                raise IndexError(
                    f"Incorrect number of indices inputted for tensor of shape {self.shape}"
                )
            for i, index in enumerate(indices):
                if index >= self.shape[i]:
                    raise IndexError(
                        f"Incorrect value for index {i}. Expected index less than {self.shape[i]}. Got {index}."
                    )
                elif index < 0:
                    raise IndexError(f"Inputted an index less than 0. Unsupported.")

        indices = (len(indices) * ctypes.c_int)(*indices)

        item = Tensor._C.get_item(self.tensor, indices)
        return item

    def __add__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_add_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_add_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_add_tensor(self.tensor, operand)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim
            result_tensor.size = 1
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

        elif isinstance(operand, Tensor):
            if self.ndim != operand.ndim:
                raise ValueError(
                    f"Inconsistent number of dimensions for tensor-tensor addition. Got {self.ndim} and {operand.ndim}."
                )

            if self.shape != operand.shape:
                raise ValueError(
                    f"Inconsistent dimensions in shape for tensor-tensor addition. Got {self.shape} and {operand.shape}."
                )

            Tensor._C.add_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.add_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.add_tensor(self.tensor, operand.tensor)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim
            result_tensor.size = 1
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

    def __sub__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_sub_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_sub_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_sub_tensor(self.tensor, operand)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndimm = self.ndim
            result_tensor.size = 1
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

        elif isinstance(operand, Tensor):
            if self.ndim != operand.ndim:
                raise ValueError(
                    f"Inconsistent number of dimensions for tensor-tensor subtraction. Got {self.ndim} and {operand.ndim}."
                )

            if self.shape != operand.shape:
                raise ValueError(
                    f"Inconsistent dimensions in shape for tensor-tensor subtraction. Got {self.shape} and {operand.shape}."
                )

            Tensor._C.sub_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.sub_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.sub_tensor(self.tensor, operand.tensor)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim
            result_tensor.size = 1
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

    def __mul__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_mul_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_mul_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_mul_tensor(self.tensor, operand)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim
            result_tensor.size = 1
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

        elif isinstance(operand, Tensor):
            if self.ndim != operand.ndim:
                raise ValueError(
                    f"Inconsistent number of dimensions for tensor-tensor hadamard multiplication. Got {self.ndim} and {operand.ndim}."
                )

            if self.shape != operand.shape:
                raise ValueError(
                    f"Inconsistent dimensions in shape for tensor-tensor hadamard multiplication. Got {self.shape} and {operand.shape}."
                )

            Tensor._C.hadamard_mul_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.hadamard_mul_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.hadamard_mul_tensor(self.tensor, operand.tensor)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim
            result_tensor.size = 1
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

    def __truediv__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_div_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_div_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_div_tensor(self.tensor, operand)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = self.shape.copy()
            result_tensor.ndim = self.ndim
            result_tensor.size = 1
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

    def __matmul__(self, operand):
        if self.ndim == 1:
            if self.shape != operand.shape:
                raise ValueError(
                    f"Incompatible shapes for inner product. Got {self.shape} and {operand.shape}."
                )

            if isinstance(operand, Tensor) and operand.ndim == 1:
                Tensor._C.inner_product_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.inner_product_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.inner_product_tensor(
                    self.tensor, operand.tensor
                )

                result_tensor = Tensor()
                result_tensor.tensor = c_result_tensor
                result_tensor.shape = 1
                result_tensor.ndim = 1
                result_tensor.size = 1

                c_result_data_ptr = c_result_tensor.contents.data
                result_tensor.data = self.__retrieve_data(
                    c_result_data_ptr, result_tensor.size
                )

                return result_tensor

            else:
                raise ValueError(
                    f"Tensors of different shapes cannot multiply. Got {self.shape} and {operand.shape}"
                )
        elif self.ndim == 2:
            if isinstance(operand, Tensor) and operand.ndim == 1:
                if self.shape[1] != operand.shape[0]:
                    raise ValueError(
                        f"Inconsistent dimensions for matmul. Got {self.shape} and {operand.shape}."
                    )
                Tensor._C.matmul_tensor_vector.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.matmul_tensor_vector.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.matmul_tensor_vector(
                    self.tensor, operand.tensor
                )

                result_tensor = Tensor()
                result_tensor.tensor = c_result_tensor
                result_tensor.shape = [self.shape[0]]
                result_tensor.ndim = 1
                result_tensor.size = self.shape[0]

                c_result_data_ptr = c_result_tensor.contents.data
                result_tensor.data = self.__retrieve_data(
                    c_result_data_ptr, result_tensor.size
                )

                return result_tensor

            elif isinstance(operand, Tensor) and operand.ndim == 2:
                if self.shape[1] != operand.shape[0]:
                    raise ValueError(
                        f"Inconsistent dimensions for matmul. Got {self.shape} and {operand.shape}."
                    )

                Tensor._C.matmul_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.matmul_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.matmul_tensor(self.tensor, operand.tensor)

                result_tensor = Tensor()
                result_tensor.tensor = c_result_tensor
                result_tensor.shape = [self.shape[0], operand.shape[1]]
                result_tensor.ndim = self.ndim
                result_tensor.size = 1
                for dim in result_tensor.shape:
                    result_tensor.size *= dim

                c_result_data_ptr = c_result_tensor.contents.data
                result_tensor.data = self.__retrieve_data(
                    c_result_data_ptr, result_tensor.size
                )

                return result_tensor

        elif self.ndim == 3:
            if operand.ndim != self.ndim:
                raise ValueError(
                    f"Inconsistent number of dimensions between batch tensor matmul operands. Got {self.ndim} and {operand.ndim}."
                )

            if operand.shape[0] != self.shape[0]:
                raise ValueError(
                    f"Inconsistent batch dimensions between batch tensor matmul operands. Got {self.shape} and {operand.shape}."
                )

            if operand.shape[1] != self.shape[2]:
                raise ValueError(
                    f"Inconsistent dimensions for matrix multiplication in batch tensor matmul. Got {self.shape} and {operand.shape}."
                )

            Tensor._C.batch_matmul_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.batch_matmul_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.batch_matmul_tensor(self.tensor, operand.tensor)

            result_tensor = Tensor()
            result_tensor.tensor = c_result_tensor
            result_tensor.shape = [self.shape[0], self.shape[1], operand.shape[2]]
            result_tensor.ndim = self.ndim
            for dim in result_tensor.shape:
                result_tensor.size *= dim

            c_result_data_ptr = c_result_tensor.contents.data
            result_tensor.data = self.__retrieve_data(
                c_result_data_ptr, result_tensor.size
            )

            return result_tensor

    def __retrieve_data(self, c_data_ptr, size_of_data: int):
        result_data = []
        for idx in range(size_of_data):
            result_data.append(c_data_ptr[idx])
        return result_data
