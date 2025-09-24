#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg.h"
#include "jetdl/math.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"

namespace jetdl {

MatVecBackward::MatVecBackward(std::shared_ptr<Tensor>& a,
                               std::shared_ptr<Tensor>& b,
                               std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> MatVecBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);
  if (tensorA->requires_grad) {
    std::shared_ptr<Tensor> unsqueezed_tensorB = unsqueeze(tensorB, 0);
    std::shared_ptr<Tensor> unsqueezed_grad_tensor = unsqueeze(grad_tensor, -1);
    std::shared_ptr<Tensor> gradA =
        linalg::matmul(unsqueezed_grad_tensor, unsqueezed_tensorB);
    grads[0] = math::sum_to_shape(gradA, tensorA->shape);
  }

  if (tensorB->requires_grad) {
    std::shared_ptr<Tensor> tensorA_mT = linalg::mT(tensorA);
    std::shared_ptr<Tensor> gradB = linalg::matmul(tensorA_mT, grad_tensor);
    grads[1] = math::sum_to_shape(gradB, tensorB->shape);
  }

  return grads;
}

}  // namespace jetdl
