#ifndef JETDL_AUTOGRAD_LINALG_H
#define JETDL_AUTOGRAD_LINALG_H

#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/tensor.h"

namespace jetdl {

class DotBackward : public Function {
 public:
  DotBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
              std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class MatVecBackward : public Function {
 public:
  MatVecBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
                 std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class VecMatBackward : public Function {
 public:
  VecMatBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
                 std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class MatmulBackward : public Function {
 public:
  MatmulBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
                 std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

}  // namespace jetdl

#endif
