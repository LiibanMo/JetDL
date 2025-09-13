#ifndef JETDL_MATH_ARITH_HPP
#define JETDL_MATH_ARITH_HPP

#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"

jetdl::Tensor _math_ops(const jetdl::Tensor& a, const jetdl::Tensor& b,
                        jetdl::utils::ArithType arith_type);
jetdl::Tensor _math_ops_a_scalar(const jetdl::Tensor& a, const jetdl::Tensor& b,
                                 jetdl::utils::ArithType arith_type);
jetdl::Tensor _math_ops_b_scalar(const jetdl::Tensor& a, const jetdl::Tensor& b,
                                 jetdl::utils::ArithType arith_type);
jetdl::Tensor _math_ops_scalars(const jetdl::Tensor& a, const jetdl::Tensor& b,
                                jetdl::utils::ArithType arith_type);

#endif
