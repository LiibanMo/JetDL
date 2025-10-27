#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/routines.h"

namespace jetdl {

ContiguousBackward::ContiguousBackward(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{a->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> ContiguousBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensor = this->saved_tensors[0];

  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);
  if (tensor->requires_grad) {
    grads[0] = grad_tensor;
  }

  return grads;
}

}  // namespace jetdl