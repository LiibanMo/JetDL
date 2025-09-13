#include "jetdl/linalg.h"

#include "jetdl/tensor.h"

namespace jetdl {

Tensor Tensor::matmul(const Tensor& other) const {
  return jetdl::linalg::matmul(*this, other);
}

Tensor Tensor::T() const { return jetdl::linalg::T(*this); }

Tensor Tensor::mT() const { return jetdl::linalg::mT(*this); }

}  // namespace jetdl
