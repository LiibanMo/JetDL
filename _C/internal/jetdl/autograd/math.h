#ifndef JETDL_AUTOGRAD_MATH_H
#define JETDL_AUTOGRAD_MATH_H

#include <memory>

#include "jetdl/autograd.h"

namespace jetdl {

class AddBackward : public Function {
 public:
  AddBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

// class SubBackward : public Function {
//  public:
//   SubBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b);
//
//   std::vector<std::shared_ptr<Tensor>> apply(
//       std::shared_ptr<Tensor>& grad_tensor) override;
// };
//
// class MulBackward : public Function {
//  public:
//   MulBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b);
//
//   std::vector<std::shared_ptr<Tensor>> apply(
//       std::shared_ptr<Tensor>& grad_tensor) override;
// };
//
// class DivBackward : public Function {
//  public:
//   DivBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b);
//
//   std::vector<std::shared_ptr<Tensor>> apply(
//       std::shared_ptr<Tensor>& grad_tensor) override;
// };

}  // namespace jetdl

#endif
