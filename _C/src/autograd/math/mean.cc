#include <memory>
#include <vector>

#include "jetdl/autograd/math.h"
#include "jetdl/math.h"
#include "jetdl/routines.h"

namespace jetdl {

MeanBackward::MeanBackward(std::shared_ptr<Tensor>& a,
                           std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{a->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> MeanBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];

  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (tensorA->requires_grad) {
    const float value = 1 / static_cast<float>(tensorA->size);
    // Create fill tensor on the same device as the parent tensor
    std::shared_ptr<Tensor> full_tensor_value =
        fill(tensorA->shape, value, false, tensorA->device);
    grads[0] = math::mul(grad_tensor, full_tensor_value);
  }

  return grads;
}

}  // namespace jetdl
