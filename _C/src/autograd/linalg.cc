#include "jetdl/autograd/linalg.h"

#include <memory>

#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

DotBackward::DotBackward(const std::shared_ptr<Tensor>& a,
                         const std::shared_ptr<Tensor>& b) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
}

std::vector<std::shared_ptr<Tensor>> DotBackward::apply(
    const std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);
  if (tensorA->requires_grad) {
    const Tensor& gradA = math::mul(*grad_tensor, *tensorB);
    grads[0] = std::make_shared<Tensor>(gradA);
  }
  if (tensorB->requires_grad) {
    const Tensor& gradB = math::mul(*grad_tensor, *tensorA);
    grads[1] = std::make_shared<Tensor>(gradB);
  }
  return grads;
}

}  // namespace jetdl
