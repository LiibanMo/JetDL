#include "tensor.hpp"

#include <cstring>

namespace py = pybind11;

std::vector<float> Tensor::_flatten_nested_pylist(py::list data) {
    std::vector<float> result;
    for (auto item : data) {
        if (py::isinstance<py::list>(item)) {
            auto nested_result = _flatten_nested_pylist(py::cast<py::list>(item));
            result.insert(result.end(), nested_result.begin(), nested_result.end());
        } else {
            result.push_back(py::cast<float>(item));
        }
    }
    return result;
}

std::vector<int> Tensor::_obtain_shape(py::list data) {
    std::vector<int> shape;
    if (data.empty()) {
        return shape;
    }   
    shape.push_back(static_cast<int>(data.size()));
    if (!data.empty() && py::isinstance<py::list>(data[0])) {
        auto nested_shape = _obtain_shape(py::cast<py::list>(data[0]));
        shape.insert(shape.end(), nested_shape.begin(), nested_shape.end());
    }
    return shape;
}

Tensor::Tensor(py::list data, bool requires_grad) {
    this->_data = this->_flatten_nested_pylist(data);
    this->shape = _obtain_shape(data);
    this->ndim = this->shape.size();
    this->size = this->_data.size();
    this->strides = std::vector<int>(this->ndim, 1);
    for (int idx = this->ndim-1; idx > 0; idx--) {
        this->strides[idx-1] = this->strides[idx] * this->shape[idx];
    }
    this->requires_grad = requires_grad;
}

Tensor::Tensor() {};