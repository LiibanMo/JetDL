#ifndef JETDL_AUTOGRAD_ROUTINES_H
#define JETDL_AUTOGRAD_ROUTINES_H

#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/tensor.h"

namespace jetdl {

class CopyBackward : public Function {
 public:
  CopyBackward(std::shared_ptr<Tensor>& a,
               std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class ViewBackward : public Function {
 public:
  ViewBackward(std::shared_ptr<Tensor>& a,
               std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class ContiguousBackward : public Function {
 public:
  ContiguousBackward(std::shared_ptr<Tensor>& a,
                     std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

}  // namespace jetdl

#endif
