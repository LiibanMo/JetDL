#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>

#include "jetdl/tensor.h"
#include "jetdl/utils/metadata.h"
#include "jetdl/utils/py.h"

namespace py = pybind11;

namespace jetdl {

Tensor::Tensor()
    : _data(nullptr),
      ndim(0),
      shape({}),
      size(0),
      strides({}),
      is_contiguous(false),
      requires_grad(false),
      grad_fn(nullptr),
      grad(nullptr) {}

Tensor::Tensor(const py::object& data, const bool requires_grad)
    : requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(nullptr),
      is_contiguous(true) {
  if (py::isinstance<py::list>(data)) {
    jetdl::utils::py_check_data_consistency(data);
    this->ndim = jetdl::utils::py_get_ndim(data);
    this->shape = jetdl::utils::py_get_shape(data, this->ndim);
    this->size = jetdl::utils::get_size(this->shape);
    this->strides = jetdl::utils::get_strides(this->shape);
    this->_data = jetdl::utils::py_flatten_list(data);
  } else if (py::isinstance<py::int_>(data) ||
             py::isinstance<py::float_>(data)) {
    this->ndim = 0;
    this->shape = {};
    this->size = 1;
    this->strides = {};
    this->_data = std::make_shared<std::vector<float>>(py::cast<float>(data));
  } else {
    throw py::type_error(
        py::str("init(): type '{}' invalid").format(py::type::of(data)));
  }
}

Tensor::Tensor(const std::shared_ptr<std::vector<float>>& data,
               const std::vector<size_t>& shape, const bool requires_grad)
    : _data(data),
      ndim(shape.size()),
      shape(shape),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(nullptr),
      is_contiguous(true) {
  this->size = jetdl::utils::get_size(this->shape);
  this->strides = jetdl::utils::get_strides(this->shape);
}

Tensor::Tensor(const float& data, const bool requires_grad)
    : ndim(0),
      shape({}),
      size(1),
      strides({}),
      is_contiguous(true),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(nullptr) {
  this->_data = std::make_shared<std::vector<float>>(1, data);
}

Tensor::Tensor(const Tensor& other)
    : _data(other._data),
      ndim(other.ndim),
      shape(other.shape),
      size(other.size),
      strides(other.strides),
      is_contiguous(other.is_contiguous),
      requires_grad(other.requires_grad),
      grad_fn(other.grad_fn),
      grad(other.grad) {}

}  // namespace jetdl
