#include <memory>
#include <stdexcept>
#include <vector>

#include "jetdl/tensor.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> Tensor::view(const std::vector<size_t>& shape) const {
  auto view_tensor = std::make_shared<Tensor>();
  view_tensor->_data = this->_data;
  view_tensor->ndim = shape.size();
  view_tensor->shape = shape;
  view_tensor->size = utils::get_size(shape);
  if (view_tensor->size != this->size) {
    throw std::logic_error("shape incompatible for creating view_tensor->\n");
  }
  view_tensor->strides = utils::get_strides(shape);
  view_tensor->requires_grad = this->requires_grad;
  view_tensor->grad_fn = nullptr;  // assigned by user after returning
  view_tensor->grad = nullptr;     // assigned by user after returning
  return view_tensor;
}

std::shared_ptr<Tensor> Tensor::squeeze(const size_t axis) const {
  if (axis >= this->ndim) {
    throw std::runtime_error("INTERNAL: squeeze(): axis >= ndim\n");
  }
  auto shape = std::vector<size_t>();
  for (size_t i = 0; i < this->shape.size(); i++) {
    const size_t dim = this->shape[i];
    if (i == axis && dim == 1) {
      continue;
    } else {
      shape.push_back(dim);
    }
  }
  return this->view(shape);
}

std::shared_ptr<Tensor> Tensor::unsqueeze(const size_t axis) const {
  if (axis >= this->ndim) {
    throw std::runtime_error("INTERNAL: unsqueeze(): axis >= ndim\n");
  }
  std::vector<size_t> shape = this->shape;
  shape.insert(shape.begin() + axis, 1);
  return this->view(shape);
}

}  // namespace jetdl
