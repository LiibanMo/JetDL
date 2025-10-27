#include <memory>

#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

Tensor Tensor::operator+(Tensor& other) {
  auto a = std::make_shared<Tensor>(*this);
  auto b = std::make_shared<Tensor>(other);
  return *math::add(a, b);
}

Tensor Tensor::operator-(Tensor& other) {
  auto a = std::make_shared<Tensor>(*this);
  auto b = std::make_shared<Tensor>(other);
  return *math::sub(a, b);
}

Tensor Tensor::operator-() {
  auto zero_tensor = std::make_shared<Tensor>(0.0f);
  auto input_tensor = std::make_shared<Tensor>(*this);
  return *math::sub(zero_tensor, input_tensor);
}

Tensor Tensor::operator*(Tensor& other) {
  auto a = std::make_shared<Tensor>(*this);
  auto b = std::make_shared<Tensor>(other);
  return *math::mul(a, b);
}

Tensor Tensor::operator/(Tensor& other) {
  auto a = std::make_shared<Tensor>(*this);
  auto b = std::make_shared<Tensor>(other);
  return *math::div(a, b);
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b) {
  return math::add(a, b);
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b) {
  return math::sub(a, b);
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor>& input) {
  auto zero_tensor = std::make_shared<Tensor>(0.0f);
  return math::sub(zero_tensor, input);
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b) {
  return math::mul(a, b);
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b) {
  return math::div(a, b);
}

}  // namespace jetdl
