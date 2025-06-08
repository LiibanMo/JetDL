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


def _C_to_Python_create_tensor(c_result_tensor):
    from .tensor import Tensor

    result_tensor = Tensor()
        
    result_tensor._tensor = c_result_tensor
    result_tensor.ndim = int(c_result_tensor.contents.ndim)
    result_tensor.size = int(c_result_tensor.contents.size)

    c_result_data_ptr = c_result_tensor.contents.data
    result_tensor._data = []
    for idx in range(result_tensor.size):
        result_tensor._data.append(c_result_data_ptr[idx])

    c_result_shape_ptr = c_result_tensor.contents.shape
    result_tensor._shape = []

    c_result_strides_ptr = c_result_tensor.contents.strides
    result_tensor.strides = []

    iterations = max(result_tensor.ndim, 1)
    for idx in range(iterations):
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
            continue
    return True


def _assign_grad_and_grad_fn(tensorA, tensorB, result_tensor, grad_fn):
    from ..autograd.control_utils import GradMode
    from .tensor import Tensor

    if isinstance(tensorB, Tensor):
        result_tensor.requires_grad = (
            tensorA.requires_grad or tensorB.requires_grad
        ) and GradMode.is_enabled()
    else:
        result_tensor.requires_grad = tensorA.requires_grad and GradMode.is_enabled()

    if result_tensor.requires_grad:
        result_tensor.grad_fn = grad_fn(tensorA, tensorB, result_tensor)


def _build_comp_graph(tensor):
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
