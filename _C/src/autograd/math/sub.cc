#include <memory>

#include "jetdl/autograd.h"
#include "jetdl/autograd/math.h"
#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

SubBackward::SubBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
                         std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> SubBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);
  if (tensorA->requires_grad) {
    grads[0] = math::sum_to_shape(grad_tensor, tensorA->shape);
  }
  if (tensorB->requires_grad) {
    std::shared_ptr<Tensor> gradB =
        math::sum_to_shape(grad_tensor, tensorB->shape);
    grads[1] = math::neg(gradB);
  }

  return grads;
}

}  // namespace jetdl
