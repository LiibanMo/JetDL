#include <memory>

#include "jetdl/tensor.h"

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

}  // namespace jetdl
