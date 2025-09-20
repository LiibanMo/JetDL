#ifndef JETDL_AUTOGRAD_H
#define JETDL_AUTOGRAD_H

#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {

class Function {
 public:
  std::vector<std::shared_ptr<Function>> next_functions = {};
  std::vector<std::shared_ptr<Tensor>> saved_tensors = {};

  virtual std::vector<std::shared_ptr<Tensor>> apply(
      const std::shared_ptr<Tensor>& grad_tensor) = 0;

  virtual ~Function() = default;
};

}  // namespace jetdl

#endif
