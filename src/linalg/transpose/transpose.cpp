#include "transpose.hpp"
#include "utils/metadata.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

Tensor c_transpose(const Tensor& tensor) {
    Tensor result_tensor = Tensor();

    result_tensor._data = std::make_shared<float[]>(tensor.size);
    std::copy(tensor._data.get(), tensor._data.get() + tensor.size, result_tensor._data.get());
    
    utils::metadata::assign_basic_metadata(result_tensor, tensor.shape);
    std::reverse(result_tensor.shape.begin(), result_tensor.shape.end());
    std::reverse(result_tensor.strides.begin(), result_tensor.strides.end());
    
    result_tensor.requires_grad = tensor.requires_grad;
    result_tensor.is_contiguous = (result_tensor.ndim < 2) ? true : false;

    return result_tensor;
}   

Tensor c_matrix_transpose(const Tensor& tensor) {
    if (tensor.ndim < 2) {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error(
            py::str(
                "tensor.mT only supports matrices or batches of matrices. Got {}-D tensor."
            ).format(tensor.ndim)
        );
    } 
    

    Tensor result_tensor = Tensor();

    result_tensor._data = std::make_shared<float[]>(tensor.size);
    std::copy(tensor._data.get(), tensor._data.get() + tensor.size, result_tensor._data.get());
    
    utils::metadata::assign_basic_metadata(result_tensor, tensor.shape);
    std::reverse(result_tensor.shape.end()-2, result_tensor.shape.end());
    std::reverse(result_tensor.strides.end()-2, result_tensor.strides.end());

    result_tensor.requires_grad = tensor.requires_grad;
    result_tensor.is_contiguous = (tensor.shape[tensor.ndim-1] == 1) ? true : false;

    return result_tensor;
}