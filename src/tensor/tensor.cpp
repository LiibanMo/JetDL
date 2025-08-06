#include "tensor.hpp"
#include "utils/metadata.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

namespace py = pybind11;

Tensor::Tensor(py::list& data, bool requires_grad) {
    this->_data = utils::metadata::flattenNestedPylist(data);
    this->shape = utils::metadata::getShape(data);
    this->ndim = utils::metadata::getNumDim(this->shape);
    this->size = utils::metadata::getSize(this->shape);
    this->strides = utils::metadata::getStrides(this->shape);

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