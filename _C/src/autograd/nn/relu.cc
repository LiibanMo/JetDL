#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/nn.h"
#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

ReLUBackward::ReLUBackward(std::shared_ptr<Tensor>& a,
                           std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{a->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> ReLUBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensor = this->saved_tensors[0];

  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);
  if (tensor->requires_grad) {
    std::shared_ptr<Tensor> local_grad = math::heaviside(tensor, 0.0f);
    grads[0] = math::mul(grad_tensor, local_grad);
  }

  return grads;
}

}  // namespace jetdl