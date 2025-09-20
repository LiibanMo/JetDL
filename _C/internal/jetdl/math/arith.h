#ifndef JETDL_MATH_ARITH_HPP
#define JETDL_MATH_ARITH_HPP

#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"

namespace jetdl {

Tensor _math_ops(const Tensor& a, const Tensor& b, utils::ArithType arith_type);
Tensor _math_ops_a_scalar(const Tensor& a, const Tensor& b,
                          utils::ArithType arith_type);
Tensor _math_ops_b_scalar(const Tensor& a, const Tensor& b,
                          utils::ArithType arith_type);
Tensor _math_ops_scalars(const Tensor& a, const Tensor& b,
                         utils::ArithType arith_type);

}  // namespace jetdl

#endif
