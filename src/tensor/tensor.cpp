#include "tensor.hpp"
#include "utils/metadata.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

namespace py = pybind11;

Tensor::Tensor(py::list& data, bool requires_grad) {
    this->_data = utils::metadata::flatten_nested_pylist(data);
    this->shape = utils::metadata::get_shape(data);
    this->ndim = utils::metadata::get_ndim(this->shape);
    this->size = utils::metadata::get_size(this->shape);
    this->strides = utils::metadata::get_strides(this->shape);

    this->requires_grad = requires_grad;
    this->grad_fn = nullptr;
    this->grad = nullptr;

    this->is_contiguous = true;
    this->is_leaf = true;
}

Tensor::Tensor(const float data, bool requires_grad) {
    this->_data = std::shared_ptr<float[]>(new float[1]);
    this->_data[0] = data;
    this->shape = {};
    this->ndim = 0;
    this-> size = 1;
    this->strides = {};

    this->requires_grad = requires_grad;
    this->grad_fn = nullptr;
    this->grad = nullptr;

    this->is_contiguous = true;
    this->is_leaf = true;
}

Tensor::Tensor() {
    this->_data = nullptr;
    this->grad_fn = nullptr;
    this->grad = nullptr;
    this->is_contiguous = true;
    this->is_leaf = false;
};