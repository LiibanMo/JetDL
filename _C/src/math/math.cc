#include "jetdl/math.h"

#include <memory>
#include <stdexcept>

#include "jetdl/math/arith.h"
#include "jetdl/math/function.h"
#include "jetdl/math/kernel.h"
#include "jetdl/tensor.h"
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

void add_inplace(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b) {
  if (a->shape != b->shape) {
    throw std::runtime_error(
        "add_inplace requires tensors with identical shapes");
  }
  if (a->device != b->device) {
    throw std::runtime_error(
        "add_inplace requires tensors on the same device");
  }

  if (a->device.is_cuda()) {
#ifdef JETDL_WITH_CUDA
    c_add_inplace_cuda(a->_cuda_data, b->_cuda_data, a->size);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    c_add_inplace_cpu(a->_data.get(), b->_data.get(), a->size);
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

std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor>& a) {
  auto zero_tensors = std::make_shared<Tensor>(0.0f);
  return sub(zero_tensors, a);
}

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor>& a, const double exponent) {
  return _power(a, exponent);
}

std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor>& a) {
  return _square_root(a);
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

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor>& a,
                             const std::vector<int>& axes) {
  if (axes.empty()) {
    return _math_total_mean(a);
  } else {
    utils::check_axes(a->shape, axes);
    std::vector<size_t> updated_axes = utils::make_axes_positive(axes, a->ndim);
    std::sort(updated_axes.begin(), updated_axes.end());
    return _math_mean_over_axes(a, updated_axes);
  }
}

std::shared_ptr<Tensor> sum_to_shape(std::shared_ptr<Tensor>& tensor,
                                     const std::vector<size_t>& shape) {
  return _math_sum_to_shape(tensor, shape);
}

std::shared_ptr<Tensor> heaviside(std::shared_ptr<Tensor>& a,
                                  const float value) {
  return _heaviside_function(a, value);
}

std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor>& a) {
  return _exp(a);
}

std::shared_ptr<Tensor> log(std::shared_ptr<Tensor>& a) {
  return _log(a);
}

std::shared_ptr<Tensor> log10(std::shared_ptr<Tensor>& a) {
  return _log10(a);
}

std::shared_ptr<Tensor> log2(std::shared_ptr<Tensor>& a) {
  return _log2(a);
}

std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor>& a) {
  return _sin(a);
}

std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor>& a) {
  return _cos(a);
}

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor>& a) {
  return _tanh(a);
}

std::shared_ptr<Tensor> sinh(std::shared_ptr<Tensor>& a) {
  return _sinh(a);
}

std::shared_ptr<Tensor> cosh(std::shared_ptr<Tensor>& a) {
  return _cosh(a);
}

std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor>& a) {
  return _abs(a);
}

std::shared_ptr<Tensor> sign(std::shared_ptr<Tensor>& a) {
  return _sign(a);
}

std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor>& a, float min_val,
                               float max_val) {
  return _clamp(a, min_val, max_val);
}

}  // namespace math
}  // namespace jetdl
