#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg.h"
#include "jetdl/math.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"

namespace jetdl {

VecMatBackward::VecMatBackward(std::shared_ptr<Tensor>& a,
                               std::shared_ptr<Tensor>& b,
                               std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> VecMatBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);

  if (tensorA->requires_grad) {
    std::shared_ptr<Tensor> tensorB_mT = linalg::mT(tensorB);
    std::shared_ptr<Tensor> gradA = linalg::matmul(grad_tensor, tensorB_mT);
    grads[0] = math::sum_to_shape(gradA, tensorA->shape);
  }

  if (tensorB->requires_grad) {
    std::shared_ptr<Tensor> unsqueezed_tensorA = unsqueeze(tensorA, -1);
    std::shared_ptr<Tensor> unsqueezed_grad_tensor = unsqueeze(grad_tensor, 0);
    std::shared_ptr<Tensor> gradB =
        linalg::matmul(unsqueezed_tensorA, unsqueezed_grad_tensor);

    grads[1] = math::sum_to_shape(gradB, tensorB->shape);
  }

  return grads;
}

}  // namespace jetdl
