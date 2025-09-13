#ifndef JETDL_MATH_REDUCTION_HPP
#define JETDL_MATH_REDUCTION_HPP

#include "jetdl/tensor.h"

jetdl::Tensor _math_total_sum(const jetdl::Tensor& a);
jetdl::Tensor _math_sum_over_axes(const jetdl::Tensor& a,
                                  const std::vector<int>& axes);

#endif
