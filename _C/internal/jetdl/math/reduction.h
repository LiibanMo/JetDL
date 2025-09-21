#ifndef JETDL_MATH_REDUCTION_HPP
#define JETDL_MATH_REDUCTION_HPP

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _math_total_sum(std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> _math_sum_over_axes(std::shared_ptr<Tensor>& a,
                                            const std::vector<size_t>& axes);

std::shared_ptr<Tensor> _math_sum_to_shape(std::shared_ptr<Tensor>& tensor,
                                           const std::vector<size_t>& shape);

}  // namespace jetdl

#endif
