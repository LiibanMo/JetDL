#ifndef JETDL_MATH_REDUCTION_HPP
#define JETDL_MATH_REDUCTION_HPP

#include "jetdl/tensor.h"

namespace jetdl {

Tensor _math_total_sum(const Tensor& a);
Tensor _math_sum_over_axes(const Tensor& a, const std::vector<size_t>& axes);

Tensor _math_sum_to_shape(const Tensor& tensor,
                          const std::vector<size_t>& shape);

}  // namespace jetdl

#endif
