#include "jetdl/autograd/math.h"

#include <memory>

#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

AddBackward::AddBackward(std::shared_ptr<Tensor>& a,
                         std::shared_ptr<Tensor>& b) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
}

std::vector<std::shared_ptr<Tensor>> AddBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);
  if (tensorA->requires_grad) {
    grads[0] = jetdl::math::sum_to_shape(grad_tensor, tensorA->shape);
  }
  if (tensorB->requires_grad) {
    grads[1] = jetdl::math::sum_to_shape(grad_tensor, tensorB->shape);
  }
  return grads;
}

// SubBackward::SubBackward(std::shared_ptr<Tensor>& a,
//                          std::shared_ptr<Tensor>& b) {
//   this->next_functions =
//       std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
//   this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
// }
//
// MulBackward::MulBackward(std::shared_ptr<Tensor>& a,
//                          std::shared_ptr<Tensor>& b) {
//   this->next_functions =
//       std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
//   this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
// }
//
// DivBackward::DivBackward(std::shared_ptr<Tensor>& a,
//                          std::shared_ptr<Tensor>& b) {
//   this->next_functions =
//       std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
//   this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
// }

}  // namespace jetdl
