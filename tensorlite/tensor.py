import ctypes
import os
from typing import Optional, Union

from .autograd import *


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

    def __init__(self, input_data: Optional[list] = None, requires_grad: bool = True):

        def _flatten(data: list) -> list:
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

        if input_data is not None:
            if isinstance(input_data, list):
                data, shape = _flatten(input_data)
            elif isinstance(input_data, (int, float)):
                data = [float(input_data)]
                shape = []
            else:
                raise TypeError(f"Invalid data type: {type(input_data)}.")

            c_data = (len(data) * ctypes.c_double)(*data)
            c_shape = (len(shape) * ctypes.c_int)(*shape)
            c_ndim = ctypes.c_int(len(shape))

            self.data = data
            self._shape = shape
            self.ndim = len(shape)

            self._tensor = Tensor._C.create_tensor(
                c_data,
                c_shape,
                c_ndim,
            )

            self.size = int(self._tensor.contents.size)

            self.strides = []
            c_strides_ptr = self._tensor.contents.strides

            for idx in range(self.ndim):
                self.strides.append(c_strides_ptr[idx])

        self.requires_grad = requires_grad
        self.grad_fn = None
        self.grad = 0.0

    def __del__(self) -> None:
        if hasattr(self, "tensor"):
            Tensor._C.free_tensor(self._tensor)
            self._tensor = None

    def __str__(self) -> str:
        def print_my_tensor(tensor: Tensor, depth: int, index: list) -> str:
            if depth == tensor.ndim - 1:
                result = ""
                for i in range(tensor._shape[depth]):
                    index[depth] = i
                    if i < tensor._shape[depth] - 1:
                        result += str(tensor[index]) + ", "
                    else:
                        result += str(tensor[index])
                return result.strip()
            else:
                result = ""
                for i in range(tensor._shape[depth]):
                    index[depth] = i
                    result += " " * 8 + "["
                    result = result.strip()
                    result += print_my_tensor(tensor, depth + 1, index) + "],"
                    if i < tensor._shape[depth] - 1:
                        result += "\n" + " " * depth
                return result.strip(",")

        index = [0] * self.ndim
        if self.ndim == 0:
            result = f"tensor({self.data[0]}"
        else:
            result = "tensor([" + print_my_tensor(self, 0, index)

        if self.grad_fn:
            self_grad_fn_str = self.grad_fn.__str__().split(" ")[0].split(".")[2]
            result += f", grad_fn=<{self_grad_fn_str}>)"
        else:
            result += "])" if self.ndim != 0 else ")"

        return result

    def __repr__(self) -> str:
        return self.__str__()

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
                    f"Incorrect number of indices inputted for tensor of shape {self._shape}"
                )
            for i, index in enumerate(indices):
                if index >= self._shape[i]:
                    raise IndexError(
                        f"Incorrect value for index {i}. Expected index less than {self._shape[i]}. Got {index}."
                    )
                elif index < 0:
                    raise IndexError(f"Inputted an index less than 0. Unsupported.")

        indices = (len(indices) * ctypes.c_int)(*indices)

        item = Tensor._C.get_item(self._tensor, indices)
        return item

    def __add__(self, operand) -> "Tensor":
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_add_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_add_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_add_tensor(self._tensor, operand)

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            _assign_grad_and_grad_fn(self, None, result_tensor, AddBackward)

            return result_tensor

        elif isinstance(operand, Tensor):
            if self._shape == operand._shape:
                Tensor._C.add_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.add_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.add_tensor(self._tensor, operand._tensor)

                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                _assign_grad_and_grad_fn(self, operand, result_tensor, AddBackward)

                return result_tensor

            else:
                if not _can_broadcast(
                    self._shape, self.ndim, operand._shape, operand.ndim
                ):
                    raise RuntimeError(
                        f"Shapes of operand tensor A {self._shape} and operand tensor B {operand._shape} are incompatible for addition broadcasting."
                    )

                Tensor._C.add_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.add_broadcasted.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.add_broadcasted(
                    self._tensor, operand._tensor
                )

                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                _assign_grad_and_grad_fn(self, operand, result_tensor, AddBackward)

                return result_tensor

        else:
            raise TypeError(
                f"Wrong data type for tensor addition. Operand datatype = {type(operand)}."
            )

    def __radd__(self, operand) -> "Tensor":
        return self + operand

    def __sub__(self, operand) -> "Tensor":
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_sub_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_sub_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_sub_tensor(self._tensor, operand)

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            _assign_grad_and_grad_fn(self, None, result_tensor, SubBackward)

            return result_tensor

        elif isinstance(operand, Tensor):
            if self._shape == operand._shape:
                Tensor._C.sub_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.sub_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.sub_tensor(self._tensor, operand._tensor)

                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                _assign_grad_and_grad_fn(self, operand, result_tensor, SubBackward)

                return result_tensor

            else:
                if not _can_broadcast(
                    self._shape, self.ndim, operand._shape, operand.ndim
                ):
                    raise RuntimeError(
                        f"Shapes of operand tensor A {self._shape} and operand tensor B {operand._shape} are incompatible for subtraction broadcasting."
                    )

                Tensor._C.sub_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.sub_broadcasted.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.sub_broadcasted(
                    self._tensor, operand._tensor
                )

                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                _assign_grad_and_grad_fn(self, operand, result_tensor, SubBackward)

                return result_tensor

        else:
            raise TypeError(
                f"Wrong data type for tensor subtraction. Operand datatype = {type(operand)}."
            )

    def __neg__(self) -> "Tensor":
        return -1 * self

    def __rsub__(self, operand) -> "Tensor":
        return -self + operand

    def __mul__(self, operand) -> "Tensor":
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_mul_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_mul_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_mul_tensor(self._tensor, operand)

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            _assign_grad_and_grad_fn(self, operand, result_tensor, MulBackward)

            return result_tensor

        elif isinstance(operand, Tensor):
            if self._shape == operand._shape:
                Tensor._C.hadamard_mul_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.hadamard_mul_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.hadamard_mul_tensor(
                    self._tensor, operand._tensor
                )

                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                _assign_grad_and_grad_fn(self, operand, result_tensor, MulBackward)

                return result_tensor

            else:
                if not _can_broadcast(
                    self._shape, self.ndim, operand._shape, operand.ndim
                ):
                    raise RuntimeError(
                        f"Shapes of operand tensor A {self._shape} and operand tensor B {operand._shape} are incompatible for multiplication broadcasting."
                    )

                Tensor._C.mul_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.mul_broadcasted.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.mul_broadcasted(
                    self._tensor, operand._tensor
                )

                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                _assign_grad_and_grad_fn(self, operand, result_tensor, MulBackward)

                return result_tensor
        else:
            raise TypeError(
                f"Wrong data type for element-wise multiplication. Operand datatype = {type(operand)}."
            )

    def __matmul__(self, operand):
        if not isinstance(operand, Tensor):
            raise TypeError(f"Operand must be of type Tensor. Got {type(operand)}.")

        if self.ndim == 1 and operand.ndim == 1:
            if self._shape != operand._shape:
                raise RuntimeError(
                    f"Tensors with shape {self._shape} and {operand._shape} cannot be matrix multiplied."
                )

            Tensor._C.vector_dot_product.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.vector_dot_product.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.vector_dot_product(
                self._tensor, operand._tensor
            )

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            _assign_grad_and_grad_fn(self, operand, result_tensor, MmBackward)

            return result_tensor

        elif self.ndim == 2 and operand.ndim == 2:
            if self._shape[1] != operand._shape[0]:
                raise RuntimeError(
                    f"Tensors with shape {self._shape} and {operand._shape} cannot be matrix multiplied."
                )

            Tensor._C.matmul_2d_2d.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matmul_2d_2d(self._tensor, operand._tensor)

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            _assign_grad_and_grad_fn(self, operand, result_tensor, MmBackward)

            return result_tensor

        elif self.ndim == 1 and operand.ndim == 2:
            if self._shape[0] != operand._shape[0]:
                raise RuntimeError(
                    f"Tensors with shape {self._shape} and {operand._shape} cannot be matrix multiplied."
                )

            new_shape = [1] + self._shape.copy()

            broadcasted_tensor = self.reshape(new_shape)

            Tensor._C.matmul_2d_2d.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matmul_2d_2d(
                broadcasted_tensor._tensor, operand._tensor
            )

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            result_shape = result_tensor._shape.copy()
            result_shape.pop(0)

            result_tensor = result_tensor.reshape(result_shape)

            _assign_grad_and_grad_fn(self, operand, result_tensor, MmBackward)

            return result_tensor

        elif self.ndim == 2 and operand.ndim == 1:
            if self._shape[1] != operand._shape[0]:
                raise RuntimeError(
                    f"Tensors with shape {self._shape} and {operand._shape} cannot be matrix multiplied."
                )

            new_shape = operand._shape.copy() + [1]

            broadcasted_tensor = operand.reshape(new_shape)

            Tensor._C.matmul_2d_2d.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_2d_2d.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matmul_2d_2d(
                self._tensor, broadcasted_tensor._tensor
            )

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            result_shape = result_tensor._shape.copy()
            result_shape.pop(1)

            result_tensor = result_tensor.reshape(result_shape)

            _assign_grad_and_grad_fn(self, operand, result_tensor, MmBackward)

            return result_tensor

        elif self.ndim > 2 or operand.ndim > 2:
            if self.ndim == 1:
                if self._shape[0] != operand._shape[-2]:
                    raise RuntimeError(
                        f"Tensors with shape {self._shape} and {operand._shape} cannot be matrix multiplied."
                    )

            if operand.ndim == 1:
                if self._shape[-1] != operand._shape[0]:
                    raise RuntimeError(
                        f"Batch tensor with matrix shape {[self._shape[-2], self._shape[-1]]} and tensor with shape {operand._shape} cannot be matrix multiplied."
                    )

            if self.ndim >= 2 and operand.ndim >= 2:
                if self._shape[-1] != operand._shape[-2]:
                    raise RuntimeError(
                        f"Batch tensors with matrix shapees {[self._shape[-2], self._shape[-1]]} and {[operand._shape[-2], operand._shape[-1]]} cannot be matrix multiplied."
                    )

            if self.ndim > 2 and operand.ndim > 2:
                if self._shape[:-3] != operand._shape[:-3]:
                    raise RuntimeError(
                        f"Batch dimensions {self._shape[:-3]} and {operand._shape[:-3]} do not match."
                    )

            Tensor._C.matmul_broadcasted.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matmul_broadcasted.restype = ctypes.POINTER(C_Tensor)

            if self.ndim == 1:
                new_shape = [1] + self._shape.copy()
                broadcasted_tensor = self.reshape(new_shape)

                c_result_tensor = Tensor._C.matmul_broadcasted(
                    broadcasted_tensor._tensor, operand._tensor
                )
                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                result_shape = result_tensor._shape.copy()
                idx_to_squeeze = result_tensor.ndim - 2
                result_shape.pop(idx_to_squeeze)

                result_tensor = result_tensor.reshape(result_shape)

                _assign_grad_and_grad_fn(self, operand, result_tensor, MmBackward)

                return result_tensor

            elif operand.ndim == 1:
                new_shape = operand._shape.copy() + [1]
                broadcasted_tensor = operand.reshape(new_shape)

                c_result_tensor = Tensor._C.matmul_broadcasted(
                    self._tensor, broadcasted_tensor._tensor
                )
                result_tensor = _C_to_Python_create_tensor(c_result_tensor)

                result_shape = result_tensor._shape.copy()
                idx_to_squeeze = result_tensor.ndim - 1
                result_shape.pop(idx_to_squeeze)

                result_tensor = result_tensor.reshape(result_shape)

                _assign_grad_and_grad_fn(self, operand, result_tensor, MmBackward)

                return result_tensor

            c_result_tensor = Tensor._C.matmul_broadcasted(
                self._tensor, operand._tensor
            )

            return _C_to_Python_create_tensor(c_result_tensor)

        else:
            raise ValueError(
                f"Invalid dimensions for matmul. Got {self.ndim} and {operand.ndim}."
            )

    def __rmul__(self, operand):
        return self * operand

    def __truediv__(self, operand):
        if isinstance(operand, (int, float)):
            Tensor._C.scalar_div_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_double,
            ]
            Tensor._C.scalar_div_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.scalar_div_tensor(self._tensor, operand)

            return _C_to_Python_create_tensor(c_result_tensor)

        else:
            if self._shape == operand._shape:
                Tensor._C.div_tensor.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.div_tensor.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.div_tensor(self._tensor, operand._tensor)

                return _C_to_Python_create_tensor(c_result_tensor)

            else:
                if not _can_broadcast(
                    self._shape, self.ndim, operand._shape, operand.ndim
                ):
                    raise RuntimeError(
                        f"Shapes of operand tensor A {self._shape} and operand tensor B {operand._shape} are incompatible for multiplication broadcasting."
                    )

                Tensor._C.div_broadcasted.argtypes = [
                    ctypes.POINTER(C_Tensor),
                    ctypes.POINTER(C_Tensor),
                ]
                Tensor._C.div_broadcasted.restype = ctypes.POINTER(C_Tensor)

                c_result_tensor = Tensor._C.div_broadcasted(
                    self._tensor, operand._tensor
                )

                return _C_to_Python_create_tensor(c_result_tensor)

    def __pow__(self, exponent: Union[int, float]):
        Tensor._C.pow_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.c_double,
        ]
        Tensor._C.pow_tensor.restype = ctypes.POINTER(C_Tensor)

        c_exponent = ctypes.c_double(exponent)
        c_result_tensor = Tensor._C.pow_tensor(self._tensor, c_exponent)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        _assign_grad_and_grad_fn(self, exponent, result_tensor, PowBackward)

        return result_tensor

    def reshape(self, new_shape: list) -> "Tensor":
        if new_shape == self._shape:
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

        _c_view_tensor = Tensor._C.reshape_tensor(self._tensor, _c_shape, _c_ndim)

        view_tensor = _C_to_Python_create_tensor(_c_view_tensor)

        return view_tensor

    def flatten(self):
        Tensor._C.flatten_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        Tensor._C.flatten_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = Tensor._C.flatten_tensor(self._tensor)

        return _C_to_Python_create_tensor(c_result_tensor)

    @property
    def shape(self) -> tuple:
        if self._shape == [0]:
            return ()
        else:
            return tuple(self._shape)

    @property
    def T(self) -> "Tensor":
        Tensor._C.transpose_tensor.argtypes = [
            ctypes.POINTER(C_Tensor),
        ]
        Tensor._C.transpose_tensor.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = Tensor._C.transpose_tensor(self._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        result_tensor.data = result_tensor.flatten().data

        return result_tensor

    @property
    def mT(self) -> "Tensor":
        if self.ndim >= 2:
            Tensor._C.matrix_transpose_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.matrix_transpose_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.matrix_transpose_tensor(self._tensor)

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            result_tensor.data = result_tensor.flatten().data

            return result_tensor
        else:
            raise RuntimeError(
                f"tensor.mT is only supported on matrices or batches of matrices. Got 1D tensor."
            )

    def sum(self, axis=None) -> "Tensor":
        if axis is None:
            Tensor._C.sum_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.sum_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.sum_tensor(self._tensor)

            return _C_to_Python_create_tensor(c_result_tensor)

        elif isinstance(axis, int):
            if axis < 0:
                c_axis = ctypes.c_int(self.ndim + axis)
            else:
                c_axis = ctypes.c_int(axis)

            Tensor._C.sum_axis_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_int,
            ]
            Tensor._C.sum_axis_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.sum_axis_tensor(self._tensor, c_axis)
            return _C_to_Python_create_tensor(c_result_tensor)

        elif isinstance(axis, list):
            axes = []
            for dim in axis:
                if dim < 0:
                    axes.append(self.ndim + dim)
                else:
                    axes.append(dim)
            axes.sort()

            Tensor._C.sum_axis_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_int,
            ]
            Tensor._C.sum_axis_tensor.restype = ctypes.POINTER(C_Tensor)

            def sum_axes_recursively(tensor: "Tensor", axes: list, idx: int):
                if idx > 0:
                    c_axis = ctypes.c_int(axes[idx])
                    c_result_tensor = Tensor._C.sum_axis_tensor(tensor._tensor, c_axis)
                    result_tensor = _C_to_Python_create_tensor(c_result_tensor)
                    return sum_axes_recursively(result_tensor, axes, idx - 1)
                else:
                    c_axis = ctypes.c_int(axes[idx])
                    c_result_tensor = Tensor._C.sum_axis_tensor(tensor._tensor, c_axis)
                    return _C_to_Python_create_tensor(c_result_tensor)

            result_tensor = sum_axes_recursively(self, axes, len(axes) - 1)

            return result_tensor

    def mean(self, axis=None) -> "Tensor":
        if axis is None:
            Tensor._C.mean_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
            ]
            Tensor._C.mean_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.mean_tensor(self._tensor)

            result_tensor = _C_to_Python_create_tensor(c_result_tensor)

            _assign_grad_and_grad_fn(self, axis, result_tensor, MeanBackward)

            return result_tensor

        elif isinstance(axis, int):
            Tensor._C.mean_axis_tensor.argtypes = [
                ctypes.POINTER(C_Tensor),
                ctypes.c_int,
            ]
            Tensor._C.mean_axis_tensor.restype = ctypes.POINTER(C_Tensor)

            c_result_tensor = Tensor._C.mean_axis_tensor(self._tensor, axis)

            return _C_to_Python_create_tensor(c_result_tensor)

    def copy(self):
        c_data = (len(self.data) * ctypes.c_double)(*self.data)
        c_shape = (len(self.shape) * ctypes.c_int)(*self.shape)
        c_ndim = ctypes.c_int(len(self.shape))

        c_result_tensor = Tensor._C.create_tensor(c_data, c_shape, c_ndim)

        return _C_to_Python_create_tensor(c_result_tensor)

    def sum_to_size(self, shape: list):
        if not _can_broadcast(self._shape, self.ndim, shape, len(shape)):
            raise RuntimeError(
                f"size {tuple(self._shape)} cannot reduce to {tuple(shape)}."
            )
        broadcasted_axes = []
        for idx in range(self.ndim):
            idx_for_new_shape = idx - self.ndim + len(shape)
            if idx_for_new_shape < 0:
                broadcasted_axes.append(idx)
            elif (
                idx_for_new_shape >= 0
                and shape[idx_for_new_shape] != self._shape[idx]
                and shape[idx_for_new_shape] == 1
            ):
                broadcasted_axes.append(idx)

        def sum_axes_recursively(tensor, broadcasted_axes, idx):
            if idx > 0:
                axis = broadcasted_axes[idx]
                result_tensor = tensor.sum(axis)
                return sum_axes_recursively(result_tensor, broadcasted_axes, idx - 1)
            else:
                axis = broadcasted_axes[idx]
                result_tensor = tensor.sum(axis)
                return result_tensor

        if broadcasted_axes:
            result_tensor = sum_axes_recursively(
                self, broadcasted_axes, len(broadcasted_axes) - 1
            )
            result_tensor = result_tensor.reshape(shape)
        else:
            result_tensor = self.copy()

        return result_tensor

    def backward(self):
        def build_comp_graph(tensor):
            topo = []
            visited = set()
            temp_stack = [tensor.grad_fn]

            while temp_stack:
                current_fn = temp_stack[-1]
                if current_fn is None:
                    temp_stack.pop()
                    continue

                if current_fn not in visited:
                    all_visited = True
                    for next_fn in current_fn.next_functions:
                        if next_fn and next_fn not in visited:
                            temp_stack.append(next_fn)
                            all_visited = False

                    if all_visited:
                        temp_stack.pop()
                        visited.add(current_fn)
                        topo.append(current_fn)

                else:
                    temp_stack.pop()

            return topo

        from .routines import ones

        self.grad = ones(self.shape)

        topo = build_comp_graph(self)

        for fn in reversed(topo):
            gradients = fn.backward()
            next_tensors = fn.next_tensors
            for grad, tensor in zip(gradients, next_tensors):
                if grad and isinstance(tensor, Tensor):
                    tensor.grad += grad


def _C_to_Python_create_tensor(c_result_tensor) -> "Tensor":
    result_tensor = Tensor()
    result_tensor._tensor = c_result_tensor
    result_tensor.ndim = int(c_result_tensor.contents.ndim)
    result_tensor.size = int(c_result_tensor.contents.size)

    c_result_data_ptr = c_result_tensor.contents.data
    result_tensor.data = []
    for idx in range(result_tensor.size):
        result_tensor.data.append(c_result_data_ptr[idx])

    c_result_shape_ptr = c_result_tensor.contents.shape
    result_tensor._shape = []

    c_result_strides_ptr = c_result_tensor.contents.strides
    result_tensor.strides = []

    for idx in range(result_tensor.ndim):
        result_tensor._shape.append(c_result_shape_ptr[idx])
        result_tensor.strides.append(c_result_strides_ptr[idx])

    return result_tensor


def _can_broadcast(shapeA, ndimA, shapeB, ndimB) -> bool:
    min_ndim = min(ndimA, ndimB)
    for idx in range(min_ndim):
        if (
            shapeA[-idx - 1] != shapeB[-idx - 1]
            and shapeA[-idx - 1] != 1
            and shapeB[-idx - 1] != 1
        ):
            return False
        else:
            return True


def _assign_grad_and_grad_fn(tensorA, tensorB, result_tensor, grad_fn):
    from .routines import GradMode
    if isinstance(tensorB, Tensor):
        result_tensor.requires_grad = (tensorA.requires_grad or tensorB.requires_grad) and GradMode.is_enabled()
    else:
        result_tensor.requires_grad = tensorA.requires_grad and GradMode.is_enabled()

    if result_tensor.requires_grad:
        result_tensor.grad_fn = grad_fn(tensorA, tensorB, result_tensor)
