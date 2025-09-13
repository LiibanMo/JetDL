#ifndef JETDL_MATH_H
#define JETDL_MATH_H

#include "jetdl/tensor.h"

namespace jetdl {
namespace math {

jetdl::Tensor add(const jetdl::Tensor& a, const jetdl::Tensor& b);
jetdl::Tensor sub(const jetdl::Tensor& a, const jetdl::Tensor& b);
jetdl::Tensor mul(const jetdl::Tensor& a, const jetdl::Tensor& b);
jetdl::Tensor div(const jetdl::Tensor& a, const jetdl::Tensor& b);
jetdl::Tensor sum(const jetdl::Tensor& a, const std::vector<int>& axes);

}  // namespace math
}  // namespace jetdl

#endif
