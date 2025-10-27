#include <memory>
#include <vector>

#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg.h"
#include "jetdl/tensor.h"

namespace jetdl {

TransposeBackward::TransposeBackward(std::shared_ptr<Tensor>& a,
                                     std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{a->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> TransposeBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensor = this->saved_tensors[0];

  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (tensor->requires_grad) {
    grads[0] = linalg::T(grad_tensor);
  }

  return grads;
}

}  // namespace jetdl
