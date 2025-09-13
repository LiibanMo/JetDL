#include "jetdl/math.h"

#include "jetdl/tensor.h"

namespace jetdl {

Tensor Tensor::operator+(const Tensor& other) const {
  return jetdl::math::add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
  return jetdl::math::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
  return jetdl::math::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
  return jetdl::math::div(*this, other);
}

}  // namespace jetdl
