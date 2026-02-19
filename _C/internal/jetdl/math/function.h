#ifndef JETDL_MATH_REDUCTION_HPP
#define JETDL_MATH_REDUCTION_HPP

#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _power(std::shared_ptr<Tensor>& input,
                               const double exponent);

std::shared_ptr<Tensor> _math_total_sum(std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> _math_sum_over_axes(std::shared_ptr<Tensor>& a,
                                            const std::vector<size_t>& axes);

std::shared_ptr<Tensor> _math_sum_to_shape(std::shared_ptr<Tensor>& tensor,
                                           const std::vector<size_t>& shape);

std::shared_ptr<Tensor> _math_total_mean(std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> _math_mean_over_axes(std::shared_ptr<Tensor>& a,
                                             const std::vector<size_t>& axes);

std::shared_ptr<Tensor> _square_root(std::shared_ptr<Tensor>& input);

std::shared_ptr<Tensor> _heaviside_function(std::shared_ptr<Tensor>& input,
                                            const float value = 0.0f);

// Transcendental functions
std::shared_ptr<Tensor> _exp(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _log(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _log10(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _log2(std::shared_ptr<Tensor>& input);

// Trigonometric functions
std::shared_ptr<Tensor> _sin(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _cos(std::shared_ptr<Tensor>& input);

// Hyperbolic functions
std::shared_ptr<Tensor> _tanh(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _sinh(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _cosh(std::shared_ptr<Tensor>& input);

// Comparison functions
std::shared_ptr<Tensor> _abs(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _sign(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> _clamp(std::shared_ptr<Tensor>& input,
                               float min_val, float max_val);

}  // namespace jetdl

#endif
