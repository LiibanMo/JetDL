#ifndef JETDL_AUTOGRAD_MATH_H
#define JETDL_AUTOGRAD_MATH_H

#include <memory>

#include "jetdl/autograd.h"

namespace jetdl {

class AddBackward : public Function {
 public:
  AddBackward(const std::shared_ptr<Tensor>& a,
              const std::shared_ptr<Tensor>& b);

  std::vector<std::shared_ptr<Tensor>> apply(
      const std::shared_ptr<Tensor>& grad_tensor) override;
};

}  // namespace jetdl

#endif
