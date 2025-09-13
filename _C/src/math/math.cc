#include "jetdl/math.h"

#include "jetdl/math/arith.h"
#include "jetdl/math/reduction.h"
#include "jetdl/utils/check.h"

namespace jetdl {
namespace math {

jetdl::Tensor add(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  jetdl::utils::check_ops_shapes(a.shape, b.shape);
  if (a.ndim > 0 && b.ndim > 0) {
    return _math_ops(a, b, jetdl::utils::ArithType::ADD);
  } else if (a.ndim == 0 && b.ndim > 0) {
    return _math_ops_a_scalar(a, b, jetdl::utils::ArithType::ADD);
  } else if (a.ndim > 0 && b.ndim == 0) {
    return _math_ops_b_scalar(a, b, jetdl::utils::ArithType::ADD);
  } else {
    return _math_ops_scalars(a, b, jetdl::utils::ArithType::ADD);
  }
}

jetdl::Tensor sub(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  jetdl::utils::check_ops_shapes(a.shape, b.shape);
  if (a.ndim > 0 && b.ndim > 0) {
    return _math_ops(a, b, jetdl::utils::ArithType::SUB);
  } else if (a.ndim == 0 && b.ndim > 0) {
    return _math_ops_a_scalar(a, b, jetdl::utils::ArithType::SUB);
  } else if (a.ndim > 0 && b.ndim == 0) {
    return _math_ops_b_scalar(a, b, jetdl::utils::ArithType::SUB);
  } else {
    return _math_ops_scalars(a, b, jetdl::utils::ArithType::SUB);
  }
}

jetdl::Tensor mul(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  jetdl::utils::check_ops_shapes(a.shape, b.shape);
  if (a.ndim > 0 && b.ndim > 0) {
    return _math_ops(a, b, jetdl::utils::ArithType::MUL);
  } else if (a.ndim == 0 && b.ndim > 0) {
    return _math_ops_a_scalar(a, b, jetdl::utils::ArithType::MUL);
  } else if (a.ndim > 0 && b.ndim == 0) {
    return _math_ops_b_scalar(a, b, jetdl::utils::ArithType::MUL);
  } else {
    return _math_ops_scalars(a, b, jetdl::utils::ArithType::MUL);
  }
}

jetdl::Tensor div(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  jetdl::utils::check_ops_shapes(a.shape, b.shape);
  if (a.ndim > 0 && b.ndim > 0) {
    return _math_ops(a, b, jetdl::utils::ArithType::DIV);
  } else if (a.ndim == 0 && b.ndim > 0) {
    return _math_ops_a_scalar(a, b, jetdl::utils::ArithType::DIV);
  } else if (a.ndim > 0 && b.ndim == 0) {
    return _math_ops_b_scalar(a, b, jetdl::utils::ArithType::DIV);
  } else {
    return _math_ops_scalars(a, b, jetdl::utils::ArithType::DIV);
  }
}

jetdl::Tensor sum(const jetdl::Tensor& a, const std::vector<int>& axes) {
  if (axes.empty()) {
    return _math_total_sum(a);
  } else {
    jetdl::utils::check_axes(a.shape, axes);
    return _math_sum_over_axes(a, axes);
  }
}

}  // namespace math
}  // namespace jetdl
