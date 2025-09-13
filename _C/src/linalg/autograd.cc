#include "jetdl/autograd.h"

#include "jetdl/linalg/autograd.h"
#include "jetdl/math.h"

namespace jetdl {
namespace linalg {

void dot_apply(jetdl::Tensor& tensor) {
  if (!tensor.grad_fn) {
    throw std::logic_error("No grad_fn assigned.\n");
  }
  jetdl::Tensor& tensorA = *tensor.grad_fn->prev_tensors[0];
  jetdl::Tensor& tensorB = *tensor.grad_fn->prev_tensors[1];

  if (tensorA.requires_grad) {
    const jetdl::Tensor& gradA = jetdl::math::mul(*tensor.grad, tensorB);
    tensorA.grad = std::make_shared<jetdl::Tensor>(gradA);
  }

  if (tensorB.requires_grad) {
    const jetdl::Tensor& gradB = jetdl::math::mul(tensorA, *tensor.grad);
    tensorB.grad = std::make_shared<jetdl::Tensor>(gradB);
  }
}

}  // namespace linalg
}  // namespace jetdl
