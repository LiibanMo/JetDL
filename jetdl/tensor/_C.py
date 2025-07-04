import ctypes
import os
from typing import Union

from ._utils import _flatten, _C_to_Python_create_tensor

class C_Lib:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "libtensor.so")
    _C = ctypes.CDLL(lib_path)

class C_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
    ]

class _TensorBase:
    C_Lib._C.create_tensor.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    C_Lib._C.create_tensor.restype = ctypes.POINTER(C_Tensor)

    def __init__(self, input_data: Union[None, int, float, list] = None, requires_grad: bool = True):
        
        if input_data is not None:
            if isinstance(input_data, list):
                data, shape = _flatten(input_data)
            elif isinstance(input_data, (int, float)):
                data = [float(input_data)]
                shape = []

            c_data = (len(data) * ctypes.c_float)(*data)
            c_shape = (len(shape) * ctypes.c_int)(*shape)
            c_ndim = ctypes.c_int(len(shape))

            self._data = data
            self._shape = shape
            self.ndim = len(shape)

            self._tensor = C_Lib._C.create_tensor(
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
        self.is_contiguous = True

    def __del__(self) -> None:
        C_Lib._C.free_tensor.argtypes = [ctypes.POINTER(C_Tensor)]
        C_Lib._C.free_tensor.restype = None
        if hasattr(self, "tensor"):
            C_Lib._C.free_tensor(self._tensor)
            self._tensor = None

    def __str__(self) -> str:
        def print_my_tensor(tensor: _TensorBase, depth: int, index: list) -> str:
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
            result = f"tensor({self._data[0]}"
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
        C_Lib._C.get_item.argtypes = [
            ctypes.POINTER(C_Tensor),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        C_Lib._C.get_item.restype = ctypes.c_float

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

        c_indices = (len(indices) * ctypes.c_int)(*indices)
        strides = self.strides.copy()
        # strides.sort(reverse=True)
        c_strides = (len(strides) * ctypes.c_int)(*strides)

        item = C_Lib._C.get_item(self._tensor, c_indices, c_strides)
        return item

    def _make_contiguous(self) -> "_TensorBase":
        if self.is_contiguous:
            return self
        
        C_Lib._C.make_contiguous.argtypes = [ctypes.POINTER(C_Tensor)]
        C_Lib._C.make_contiguous.restype = ctypes.POINTER(C_Tensor)

        c_result_tensor = C_Lib._C.make_contiguous(self._tensor)

        result_tensor = _C_to_Python_create_tensor(c_result_tensor)

        self.is_contiguous = True

        return result_tensor