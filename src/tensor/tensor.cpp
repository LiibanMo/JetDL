#include "tensor.hpp"
#include "../utils/metadata.hpp"

#include <cstring>
#include <vector>

namespace py = pybind11;

Tensor::Tensor(py::list data, bool requires_grad) {
    this->_data = utils::metadata::flattenNestedPylist(data);
    this->shape = utils::metadata::getShape(data);
    this->ndim = utils::metadata::getNumDim(this->shape);
    this->size = utils::metadata::getSize(this->shape);
    this->strides = utils::metadata::getStrides(this->shape);
    this->requires_grad = requires_grad;
    this->is_contiguous = true;
}

Tensor::Tensor(float data, bool requires_grad) {
    this->_data = std::vector<float> {data};
    this->shape = std::vector<int>();
    this->ndim = 0;
    this->size = 1;
    this->strides = std::vector<int>();
    this->requires_grad = requires_grad;
    this->is_contiguous = true;
}

Tensor::Tensor() {
    this->is_contiguous = true;
};