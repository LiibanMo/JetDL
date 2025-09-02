#include "linalg.h"
#include "linalg/transpose/transpose.h"
#include "product/matmul.h"
#include "python/utils/check.h"
#include "tensor/python/bindings.h"
#include "tensor/tensor.h"

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <stdio.h>

namespace py = pybind11;

std::unique_ptr<Tensor, TensorDeleter> linalg_dot(const Tensor &a,
                                                  const Tensor &b) {
  utils_check_dot_shapes(a.shape, a.ndim, b.shape, b.ndim);
  Tensor *result_tensor = c_linalg_dot(&a, &b);
  return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
}

std::unique_ptr<Tensor, TensorDeleter> linalg_matmul(const Tensor &a,
                                                     const Tensor &b) {
  if (a.ndim == 1 && b.ndim == 1) {
    utils_check_dot_shapes(a.shape, a.ndim, b.shape, b.ndim);
    Tensor *result_tensor = c_linalg_dot(&a, &b);
    return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

  } else if (a.ndim > 1 && b.ndim == 1) {
    utils_check_matvec_shapes(a.shape, a.ndim, b.shape, b.ndim);
    Tensor *result_tensor = c_linalg_matvec(&a, &b);
    return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

  } else if (a.ndim == 1 && b.ndim > 1) {
    utils_check_vecmat_shapes(a.shape, a.ndim, b.shape, b.ndim);
    Tensor *result_tensor = c_linalg_vecmat(&a, &b);
    return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);

  } else {
    utils_check_matmul_shapes(a.shape, a.ndim, b.shape, b.ndim);
    Tensor *result_tensor = c_linalg_matmul(&a, &b);
    return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
  }
}

std::unique_ptr<Tensor, TensorDeleter> linalg_T(const Tensor &a) {
  Tensor *result_tensor = c_linalg_T(&a);
  return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
}

std::unique_ptr<Tensor, TensorDeleter> linalg_mT(const Tensor &a) {
  if (a.ndim < 2) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(py::str("tensor.mT only supports matrices or "
                                     "batches of matrices. Got {}D tensor.")
                                 .format(a.ndim));
  }
  Tensor *result_tensor = c_linalg_mT(&a);
  return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
}
