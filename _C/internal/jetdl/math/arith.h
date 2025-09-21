#ifndef JETDL_MATH_ARITH_HPP
#define JETDL_MATH_ARITH_HPP

#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"

namespace jetdl {

std::shared_ptr<Tensor> _math_ops(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b,
                                  utils::ArithType arith_type);
std::shared_ptr<Tensor> _math_ops_a_scalar(std::shared_ptr<Tensor>& a,
                                           std::shared_ptr<Tensor>& b,
                                           utils::ArithType arith_type);
std::shared_ptr<Tensor> _math_ops_b_scalar(std::shared_ptr<Tensor>& a,
                                           std::shared_ptr<Tensor>& b,
                                           utils::ArithType arith_type);
std::shared_ptr<Tensor> _math_ops_scalars(std::shared_ptr<Tensor>& a,
                                          std::shared_ptr<Tensor>& b,
                                          utils::ArithType arith_type);

}  // namespace jetdl

#endif
