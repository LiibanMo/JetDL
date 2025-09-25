#include <unistd.h>

#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/math.h"
#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

PowBackward::PowBackward(std::shared_ptr<Tensor>& a, const int exponent,
                         std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{a->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a};
  this->tensor = result_tensor;
  this->exponent = exponent;
}

std::vector<std::shared_ptr<Tensor>> PowBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];

  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (tensorA->requires_grad) {
    std::shared_ptr<Tensor> dC_dA_partial =
        math::pow(tensorA, this->exponent - 1);
    std::shared_ptr<Tensor> exponent_tensor =
        std::make_shared<Tensor>(this->exponent);
    std::shared_ptr<Tensor> dC_dA = math::mul(exponent_tensor, dC_dA_partial);
    std::shared_ptr<Tensor> gradA = math::mul(dC_dA, grad_tensor);
    grads[0] = gradA;
  }

  return grads;
}

}  // namespace jetdl
