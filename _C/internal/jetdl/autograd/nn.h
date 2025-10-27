#ifndef JETDL_AUTOGRAD_NN_H
#define JETDL_AUTOGRAD_NN_H

#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/tensor.h"
namespace jetdl {

class ReLUBackward : public Function {
 public:
  ReLUBackward(std::shared_ptr<Tensor>& a,
               std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

}  // namespace jetdl

#endif