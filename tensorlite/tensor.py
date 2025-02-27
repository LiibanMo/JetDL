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

    def __init__(self, data: Optional[list] = None):
        def __flatten(data: list) -> list:
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
        
        if data is not None:
            data, shape = __flatten(data)

            self._c_data = (len(data) * ctypes.c_double)(*data)
            self._c_shape = (len(shape) * ctypes.c_int)(*shape)
            self._c_ndim = ctypes.c_int(len(shape))

            self.data = data
            self.shape = shape
            self.ndim = len(shape)

            self.tensor = Tensor._C.create_tensor(
                self._c_data,
                self._c_shape,
                self._c_ndim,
            )

            self.size = int(self.tensor.contents.size)

            self.strides = []
            c_strides_ptr = self.tensor.contents.strides

            for idx in range(self.ndim):
                self.strides.append(c_strides_ptr[idx])

    def __del__(self) -> None:
        if hasattr(self, "tensor"):
            Tensor._C.free_tensor(self.tensor)
            self.tensor = None

    def __str__(self) -> str:
        def print_my_tensor(tensor:Tensor, depth:int, index:list) -> str:
            if depth == tensor.ndim - 1:
                result = ""
                for i in range(tensor.shape[depth]):
                    index[depth] = i
                    if i < tensor.shape[depth] - 1:
                        result += str(tensor[index]) + ", "
                    else:
                        result += str(tensor[index])
                return result.strip()
            else:
                result = ""
                for i in range(tensor.shape[depth]):
                    index[depth] = i
                    result += " " * 8 + "["
                    result = result.strip()
                    result += print_my_tensor(tensor, depth+1, index) + "],"
                    if i < tensor.shape[depth] - 1:
                        result += "\n" + " " * depth
                return result.strip(",")

        result = "tensor(["
        index = [0] * self.ndim
        result += print_my_tensor(self, 0, index) + "])"

        return result


    def __getitem__(self, indices):
        Tensor._C.get_item.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(ctypes.c_int),
        ]
        Tensor._C.get_item.restype = ctypes.c_double

        if isinstance(indices, (int, float)):
            indices = (indices,)

        if isinstance(indices, tuple):
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

            return self._C_to_Python_create_tensor(c_result_tensor)

        elif isinstance(operand, Tensor):
            if self.shape == operand.shape:
                Tensor._C.add_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.add_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.add_tensor(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)

            else:
                max_ndim = max(self.ndim, operand.ndim)
                for idx in range(max_ndim):
                    if idx < self.ndim and idx < operand.ndim:
                        if (
                            self.shape[-idx - 1] != operand.shape[-idx - 1]
                            and self.shape[-idx - 1] != 1
                            and operand.shape[-idx - 1] != 1
                        ):
                            raise RuntimeError(
                                f"Shapes of operand tensor A {self.shape} and operand tensor B {operand.shape} are incompatible for addition broadcasting."
                            )

                Tensor._C.add_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.add_broadcasted.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.add_broadcasted(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)

        else:
            raise TypeError(
                f"Wrong data type for tensor addition. Operand datatype = {type(operand)}."
            )

    def __sub__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_sub_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_sub_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_sub_tensor(self.tensor, operand)

            return self._C_to_Python_create_tensor(c_result_tensor)

        elif isinstance(operand, Tensor):
            if self.shape == operand.shape:
                Tensor._C.sub_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.sub_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.sub_tensor(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)

            else:
                max_ndim = max(self.ndim, operand.ndim)
                for idx in range(max_ndim):
                    if idx < self.ndim and idx < operand.ndim:
                        if (
                            self.shape[-idx - 1] != operand.shape[-idx - 1]
                            and self.shape[-idx - 1] != 1
                            and operand.shape[-idx - 1] != 1
                        ):
                            raise RuntimeError(
                                f"Shapes of operand tensor A {self.shape} and operand tensor B {operand.shape} are incompatible for subtraction broadcasting."
                            )

                Tensor._C.sub_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.sub_broadcasted.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.sub_broadcasted(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)
        else:
            raise TypeError(
                f"Wrong data type for tensor subtraction. Operand datatype = {type(operand)}."
            )

    def __mul__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_mul_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_mul_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_mul_tensor(self.tensor, operand)

            return self._C_to_Python_create_tensor(c_result_tensor)

        elif isinstance(operand, Tensor):
            if self.shape == operand.shape:
                Tensor._C.hadamard_mul_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.hadamard_mul_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.hadamard_mul_tensor(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)

            else:
                max_ndim = max(self.ndim, operand.ndim)
                for idx in range(max_ndim):
                    if idx < self.ndim and idx < operand.ndim:
                        if (
                            self.shape[-idx - 1] != operand.shape[-idx - 1]
                            and self.shape[-idx - 1] != 1
                            and operand.shape[-idx - 1] != 1
                        ):
                            raise RuntimeError(
                                f"Shapes of operand tensor A {self.shape} and operand tensor B {operand.shape} are incompatible for multiplication broadcasting."
                            )
                        
                Tensor._C.mul_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.mul_broadcasted.restype = ctypes.POINTER(C_Tensor)
                
                c_result_tensor = Tensor._C.mul_broadcasted(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)
        else:
            raise TypeError(
                f"Wrong data type for element-wise multiplication. Operand datatype = {type(operand)}."
            )

    def __matmul__(self, operand):
        if not isinstance(operand, Tensor):
            raise TypeError(f"Operand must be of type Tensor. Got {type(operand)}.")

        if self.ndim == 1 and operand.ndim == 1:
            if self.shape != operand.shape:
                raise RuntimeError(
                    f"Tensors with shape {self.shape} and {operand.shape} cannot be matrix multiplied."
                )

            Tensor._C.vector_dot_product.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.vector_dot_product.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.vector_dot_product(self.tensor, operand.tensor)

            result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

            return result_tensor

        elif self.ndim == 2 and operand.ndim == 2:
            if self.shape[1] != operand.shape[0]:
                raise RuntimeError(
                    f"Tensors with shape {self.shape} and {operand.shape} cannot be matrix multiplied."
                )

            Tensor._C.matmul_2d_2d.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matmul_2d_2d(self.tensor, operand.tensor)

            result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

            return result_tensor

        elif self.ndim == 1 and operand.ndim == 2:
            if self.shape[0] != operand.shape[0]:
                raise RuntimeError(
                    f"Tensors with shape {self.shape} and {operand.shape} cannot be matrix multiplied."
                )

            new_shape = [1] + self.shape.copy()

            broadcasted_tensor = self.reshape(new_shape)

            Tensor._C.matmul_2d_2d.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matmul_2d_2d(
                broadcasted_tensor.tensor, operand.tensor
            )

            result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

            result_shape = result_tensor.shape.copy()
            result_shape.pop(0)

            return result_tensor.reshape(result_shape)

        elif self.ndim == 2 and operand.ndim == 1:
            if self.shape[1] != operand.shape[0]:
                raise RuntimeError(
                    f"Tensors with shape {self.shape} and {operand.shape} cannot be matrix multiplied."
                )

            new_shape = operand.shape.copy() + [1]

            broadcasted_tensor = operand.reshape(new_shape)

            Tensor._C.matmul_2d_2d.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matmul_2d_2d(
                self.tensor, broadcasted_tensor.tensor
            )

            result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

            result_shape = result_tensor.shape.copy()
            result_shape.pop(1)

            return result_tensor.reshape(result_shape)

        elif self.ndim > 2 or operand.ndim > 2:
            if self.ndim == 1:
                if self.shape[0] != operand.shape[-2]:
                    raise RuntimeError(
                        f"Tensors with shape {self.shape} and {operand.shape} cannot be matrix multiplied."
                    )

            if operand.ndim == 1:
                if self.shape[-1] != operand.shape[0]:
                    raise RuntimeError(
                        f"Batch tensor with matrix shape {[self.shape[-2], self.shape[-1]]} and tensor with shape {operand.shape} cannot be matrix multiplied."
                    )

            if self.ndim >= 2 and operand.ndim >= 2:
                if self.shape[-1] != operand.shape[-2]:
                    raise RuntimeError(
                        f"Batch tensors with matrix shapees {[self.shape[-2], self.shape[-1]]} and {[operand.shape[-2], operand.shape[-1]]} cannot be matrix multiplied."
                    )

            if self.ndim > 2 and operand.ndim > 2:
                if self.shape[:-3] != operand.shape[:-3]:
                    raise RuntimeError(
                        f"Batch dimensions {self.shape[:-3]} and {operand.shape[:-3]} do not match."
                    )

            Tensor._C.matmul_broadcasted.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_broadcasted.restype = ctypes.POINTER(C_Tensor)

            if self.ndim == 1:
                new_shape = [1] + self.shape.copy()
                broadcasted_tensor = self.reshape(new_shape)

                c_result_tensor = Tensor._C.matmul_broadcasted(
                    broadcasted_tensor.tensor, operand.tensor
                )
                result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

                result_shape = result_tensor.shape.copy()
                idx_to_squeeze = result_tensor.ndim - 2
                result_shape.pop(idx_to_squeeze)

                return result_tensor.reshape(result_shape)

            elif operand.ndim == 1:
                new_shape = operand.shape.copy() + [1]
                broadcasted_tensor = operand.reshape(new_shape)

                c_result_tensor = Tensor._C.matmul_broadcasted(
                    self.tensor, broadcasted_tensor.tensor
                )
                result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

                result_shape = result_tensor.shape.copy()
                idx_to_squeeze = result_tensor.ndim - 1
                result_shape.pop(idx_to_squeeze)

                return result_tensor.reshape(result_shape)

            c_result_tensor = Tensor._C.matmul_broadcasted(self.tensor, operand.tensor)

            return self._C_to_Python_create_tensor(c_result_tensor)

        else:
            raise ValueError(
                f"Invalid dimensions for matmul. Got {self.ndim} and {operand.ndim}."
            )

    def __truediv__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_div_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_div_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_div_tensor(self.tensor, operand)

            return self._C_to_Python_create_tensor(c_result_tensor)

        else:
            if self.shape == operand.shape:
                Tensor._C.div_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.div_tensor.restype = ctypes.POINTER(C_Tensor)
            
                c_result_tensor = Tensor._C.div_tensor(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)
            
            else:
                max_ndim = max(self.ndim, operand.ndim)
                for idx in range(max_ndim):
                    if idx < self.ndim and idx < operand.ndim:
                        if (
                            self.shape[-idx - 1] != operand.shape[-idx - 1]
                            and self.shape[-idx - 1] != 1
                            and operand.shape[-idx - 1] != 1
                        ):
                            raise RuntimeError(
                                f"Shapes of operand tensor A {self.shape} and operand tensor B {operand.shape} are incompatible for multiplication broadcasting."
                            )

                Tensor._C.div_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.div_broadcasted.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.div_broadcasted(self.tensor, operand.tensor)

                return self._C_to_Python_create_tensor(c_result_tensor)

    def reshape(self, new_shape: list) -> "Tensor":
        if new_shape == self.shape:
            return self

        new_size = 1
        for dim in new_shape:
            new_size *= dim
        if new_size != self.size:
            raise ValueError(
                f"Reshaped tensor must be the same size as the current tensor. Got new size = {new_size}. Expected size = {self.size}."
            )

        Tensor._C.reshape_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        Tensor._C.reshape_tensor.restype = ctypes.POINTER(C_Tensor)

        _c_shape = (len(new_shape) * ctypes.c_int)(*new_shape)
        _c_ndim = ctypes.c_int(len(new_shape))

        _c_view_tensor = Tensor._C.reshape_tensor(self.tensor, _c_shape, _c_ndim)

        view_tensor = self._C_to_Python_create_tensor(_c_view_tensor)

        return view_tensor
    
    def flatten(self):
        Tensor._C.flatten_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        Tensor._C.flatten_tensor.restype = ctypes.POINTER(C_Tensor)
        
        c_result_tensor = Tensor._C.flatten_tensor(self.tensor)

        return self._C_to_Python_create_tensor(c_result_tensor)

    @property
    def T(self) -> "Tensor":
        Tensor._C.transpose_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        Tensor._C.transpose_tensor.restype = ctypes.POINTER(C_Tensor)
        
        c_result_tensor = Tensor._C.transpose_tensor(self.tensor)

        result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

        result_tensor.data = result_tensor.flatten().data

        return result_tensor

    @property
    def mT(self) -> "Tensor":
        if self.ndim >= 2:
            Tensor._C.matrix_transpose_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matrix_transpose_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matrix_transpose_tensor(self.tensor)

            result_tensor = self._C_to_Python_create_tensor(c_result_tensor)

            result_tensor.data = result_tensor.flatten().data

            return result_tensor
        else:
            raise RuntimeError(f"tensor.mT is only supported on matrices or batches of matrices. Got 1D tensor.")
        

    def sum(self, axis=None) -> "Tensor":
        if axis is None:
            Tensor._C.sum_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.sum_tensor.restype = ctypes.POINTER(C_Tensor)
            
            c_result_tensor = Tensor._C.sum_tensor(self.tensor)

        else:
            Tensor._C.sum_axis_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_int,
            ]
            Tensor._C.sum_axis_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.sum_axis_tensor(self.tensor, axis)

        return self._C_to_Python_create_tensor(c_result_tensor)

    def _C_to_Python_create_tensor(self, c_result_tensor) -> "Tensor":
        result_tensor = Tensor()
        result_tensor.tensor = c_result_tensor
        result_tensor.ndim = int(c_result_tensor.contents.ndim)
        result_tensor.size = int(c_result_tensor.contents.size)

        c_result_data_ptr = c_result_tensor.contents.data
        result_tensor.data = []
        for idx in range(result_tensor.size):
            result_tensor.data.append(c_result_data_ptr[idx])

        c_result_shape_ptr = c_result_tensor.contents.shape
        result_tensor.shape = []

        c_result_strides_ptr = c_result_tensor.contents.strides
        result_tensor.strides = []

        for idx in range(result_tensor.ndim):
            result_tensor.shape.append(c_result_shape_ptr[idx])
            result_tensor.strides.append(c_result_strides_ptr[idx])

        return result_tensor
