#include "jetdl/math.h"

#include <memory>

#include "jetdl/math/arith.h"
#include "jetdl/math/reduction.h"
#include "jetdl/utils/check.h"

namespace jetdl {
namespace math {

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b) {
  utils::check_ops_shapes(a->shape, b->shape);
  if (a->ndim > 0 && b->ndim > 0) {
    return _math_ops(a, b, utils::ArithType::ADD);
  } else if (a->ndim == 0 && b->ndim > 0) {
    return _math_ops_a_scalar(a, b, utils::ArithType::ADD);
  } else if (a->ndim > 0 && b->ndim == 0) {
    return _math_ops_b_scalar(a, b, utils::ArithType::ADD);
  } else {
    return _math_ops_scalars(a, b, utils::ArithType::ADD);
  }
}

std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b) {
  utils::check_ops_shapes(a->shape, b->shape);
  if (a->ndim > 0 && b->ndim > 0) {
    return _math_ops(a, b, utils::ArithType::SUB);
  } else if (a->ndim == 0 && b->ndim > 0) {
    return _math_ops_a_scalar(a, b, utils::ArithType::SUB);
  } else if (a->ndim > 0 && b->ndim == 0) {
    return _math_ops_b_scalar(a, b, utils::ArithType::SUB);
  } else {
    return _math_ops_scalars(a, b, utils::ArithType::SUB);
  }
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b) {
  utils::check_ops_shapes(a->shape, b->shape);
  if (a->ndim > 0 && b->ndim > 0) {
    return _math_ops(a, b, utils::ArithType::MUL);
  } else if (a->ndim == 0 && b->ndim > 0) {
    return _math_ops_a_scalar(a, b, utils::ArithType::MUL);
  } else if (a->ndim > 0 && b->ndim == 0) {
    return _math_ops_b_scalar(a, b, utils::ArithType::MUL);
  } else {
    return _math_ops_scalars(a, b, utils::ArithType::MUL);
  }
}

std::shared_ptr<Tensor> div(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b) {
  utils::check_ops_shapes(a->shape, b->shape);
  if (a->ndim > 0 && b->ndim > 0) {
    return _math_ops(a, b, utils::ArithType::DIV);
  } else if (a->ndim == 0 && b->ndim > 0) {
    return _math_ops_a_scalar(a, b, utils::ArithType::DIV);
  } else if (a->ndim > 0 && b->ndim == 0) {
    return _math_ops_b_scalar(a, b, utils::ArithType::DIV);
  } else {
    return _math_ops_scalars(a, b, utils::ArithType::DIV);
  }
}

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor>& a,
                            const std::vector<int>& axes) {
  if (axes.empty()) {
    return _math_total_sum(a);
  } else {
    utils::check_axes(a->shape, axes);
    std::vector<size_t> updated_axes = utils::make_axes_positive(axes, a->ndim);
    std::sort(updated_axes.begin(), updated_axes.end());
    return _math_sum_over_axes(a, updated_axes);
  }
}

std::shared_ptr<Tensor> sum_to_shape(std::shared_ptr<Tensor>& tensor,
                                     const std::vector<size_t>& shape) {
  return _math_sum_to_shape(tensor, shape);
}

}  // namespace math
}  // namespace jetdl
