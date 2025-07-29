#include "transpose.hpp"

#include <algorithm>
#include <stdexcept>

Tensor c_transpose(const Tensor& a) {
    Tensor result = Tensor();

    result._data = std::vector<float>(a.size, 0.0f);
    std::copy(a._data.begin(), a._data.end(), result._data.begin());
    
    result.shape = std::vector<int>(a.ndim, 0);
    std::copy(a.shape.begin(), a.shape.end(), result.shape.begin());
    std::reverse(result.shape.begin(), result.shape.end());
    
    result.ndim = a.ndim;
    result.size = a.size;
    
    result.strides = std::vector<int>(a.ndim, 0);
    std::copy(a.strides.begin(), a.strides.end(), result.strides.begin());
    std::reverse(result.strides.begin(), result.strides.end());

    result.requires_grad = a.requires_grad;
    result.is_contiguous = false;

    return result;
}   

Tensor c_matrix_transpose(const Tensor& a) {
    if (a.ndim < 2) {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error(
            py::str(
                "tensor.mT only supports matrices or batches of matrices. Got {}-D tensor."
            ).format(a.ndim)
        );
    } 

    Tensor result = Tensor();

    result._data = std::vector<float>(a.size, 0.0f);
    std::copy(a._data.begin(), a._data.end(), result._data.begin());
    
    result.shape = std::vector<int>(a.ndim, 0);
    std::copy(a.shape.begin(), a.shape.end(), result.shape.begin());
    std::reverse(result.shape.end()-2, result.shape.end());
    
    result.ndim = a.ndim;
    result.size = a.size;
    
    result.strides = std::vector<int>(a.ndim, 0);
    std::copy(a.strides.begin(), a.strides.end(), result.strides.begin());
    std::reverse(result.strides.end()-2, result.strides.end());

    result.requires_grad = a.requires_grad;
    result.is_contiguous = false;

    return result;
}