#ifndef JETDL_AUTOGRAD_LINALG_H
#define JETDL_AUTOGRAD_LINALG_H

#include "jetdl/autograd.h"
#include "jetdl/tensor.h"

namespace jetdl {

class DotBackward : public Function {
 public:
  DotBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

}  // namespace jetdl

#endif
