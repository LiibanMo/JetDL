#include "math.h"
#include "math/functions/reduction.h"
#include "math/ops/arith.h"
#include "python/utils/check.h"
#include "tensor/python/bindings.h"
#include "tensor/tensor.h"
#include "utils/auxiliary.h"

#include <memory>
#include <stdexcept>
#include <stdlib.h>

std::unique_ptr<Tensor, TensorDeleter>
math_ops(const Tensor &a, const Tensor &b, const std::string op) {
  utils_check_ops_shapes(a.shape, a.ndim, b.shape, b.ndim);
  if (op == "ADD") {
    if (a.ndim > 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops(&a, &b, ADD);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim == 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops_a_scalar(&a, &b, ADD);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim > 0 && b.ndim == 0) {
      Tensor *result_tensor = c_math_ops_b_scalar(&a, &b, ADD);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else {
      Tensor *result_tensor = c_math_ops_scalars(&a, &b, ADD);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
    }

  } else if (op == "SUB") {
    if (a.ndim > 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops(&a, &b, SUB);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim == 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops_a_scalar(&a, &b, SUB);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim > 0 && b.ndim == 0) {
      Tensor *result_tensor = c_math_ops_b_scalar(&a, &b, SUB);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else {
      Tensor *result_tensor = c_math_ops_scalars(&a, &b, SUB);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
    }

  } else if (op == "MUL") {
    if (a.ndim > 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops(&a, &b, MUL);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim == 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops_a_scalar(&a, &b, MUL);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim > 0 && b.ndim == 0) {
      Tensor *result_tensor = c_math_ops_b_scalar(&a, &b, MUL);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else {
      Tensor *result_tensor = c_math_ops_scalars(&a, &b, MUL);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
    }

  } else if (op == "DIV") {
    if (a.ndim > 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops(&a, &b, DIV);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim == 0 && b.ndim > 0) {
      Tensor *result_tensor = c_math_ops_a_scalar(&a, &b, DIV);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else if (a.ndim > 0 && b.ndim == 0) {
      Tensor *result_tensor = c_math_ops_b_scalar(&a, &b, DIV);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

    } else {
      Tensor *result_tensor = c_math_ops_scalars(&a, &b, DIV);
      return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
    }

  } else {
    throw std::logic_error(
        "ops can only of the following options: ADD, SUB, MUL, DIV.");
  }
}

std::unique_ptr<Tensor, TensorDeleter> math_sum(const Tensor &a,
                                                std::vector<int> &axes) {
  if (axes.empty()) {
    Tensor *result_tensor = c_math_total_sum(&a);
    return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

  } else {
    utils_check_axes(a.shape, a.ndim, axes.data(), axes.size());
    Tensor *result_tensor = c_math_sum_over_axes(&a, axes.data(), axes.size());
    return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
  }
}
