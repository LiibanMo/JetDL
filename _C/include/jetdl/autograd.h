#ifndef JETDL_AUTOGRAD_HPP
#define JETDL_AUTOGRAD_HPP

#include "jetdl/tensor.h"

class Function {
 public:
  std::vector<std::shared_ptr<jetdl::Tensor>> prev_tensors;
  std::weak_ptr<jetdl::Tensor> tensor;

  virtual void apply(const jetdl::Tensor& tensor) const = 0;

  virtual ~Function() {};
};

#endif
