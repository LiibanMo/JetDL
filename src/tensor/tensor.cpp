#include "tensor.hpp"
#include "../utils/metadata.hpp"

#include <cstring>

namespace py = pybind11;

Tensor::Tensor(py::list data, bool requires_grad) {
    this->_data = utils::metadata::flattenNestedPylist(data);
    this->shape = utils::metadata::getShape(data);
    this->ndim = utils::metadata::getNumDim(this->shape);
    this->size = utils::metadata::getSize(this->shape);
    this->strides = utils::metadata::getStrides(this->shape);
    this->requires_grad = requires_grad;
}

Tensor::Tensor() {};