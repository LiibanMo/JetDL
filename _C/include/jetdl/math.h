#ifndef JETDL_MATH_H
#define JETDL_MATH_H

#include "jetdl/tensor.h"

namespace jetdl {
namespace math {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor sum(const Tensor& a, const std::vector<int>& axes);
Tensor sum_to_shape(const Tensor& tensor, const std::vector<size_t>& shape);

}  // namespace math
}  // namespace jetdl

#endif
