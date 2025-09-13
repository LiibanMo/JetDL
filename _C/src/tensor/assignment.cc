#include "jetdl/tensor.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

Tensor Tensor::operator=(const Tensor& other) {
  if (this == &other) {
    return *this;
  }

  this->_data = other._data;
  this->ndim = other.ndim;
  this->shape = other.shape;
  this->size = other.size;
  this->strides = other.strides;
  this->is_contiguous = other.is_contiguous;
  this->requires_grad = other.requires_grad;
  this->grad_fn = other.grad_fn;
  this->grad = other.grad;

  return *this;
}

Tensor Tensor::view(const std::vector<size_t>& shape) const {
  Tensor view_tensor;
  view_tensor._data = this->_data;
  view_tensor.ndim = shape.size();
  view_tensor.shape = shape;
  view_tensor.size = jetdl::utils::get_size(shape);
  if (view_tensor.size != this->size) {
    throw std::logic_error("shape incompatible for creating view_tensor.\n");
  }
  view_tensor.strides = jetdl::utils::get_strides(shape);
  view_tensor.requires_grad = this->requires_grad;
  view_tensor.grad_fn = nullptr;  // assigned by user after returning
  view_tensor.grad = nullptr;     // assigned by user after returning
  return view_tensor;
}

}  // namespace jetdl
