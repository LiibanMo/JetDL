#include <memory>
#include <vector>

#include "jetdl/autograd/routines.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"

namespace jetdl {

ViewBackward::ViewBackward(std::shared_ptr<Tensor>& a,
                           std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{a->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> ViewBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];

  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (tensorA->requires_grad) {
    grads[0] = view(grad_tensor, tensorA->shape);
  }

  return grads;
}

}  // namespace jetdl
