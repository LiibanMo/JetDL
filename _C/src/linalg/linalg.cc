#include "jetdl/linalg.h"

#include <pybind11/pybind11.h>

#include <stdexcept>

#include "jetdl/linalg/product.h"
#include "jetdl/linalg/transpose.h"
#include "jetdl/utils/check.h"

namespace py = pybind11;

namespace jetdl {
namespace linalg {

jetdl::Tensor dot(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  jetdl::utils::check_dot_shapes(a.shape, b.shape);
  return _linalg_dot(a, b);
}

jetdl::Tensor matmul(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  if (a.ndim == 1 && b.ndim == 1) {
    jetdl::utils::check_dot_shapes(a.shape, b.shape);
    return _linalg_dot(a, b);
  } else if (a.ndim > 1 && b.ndim == 1) {
    jetdl::utils::check_matvec_shapes(a.shape, b.shape);
    return _linalg_matvec(a, b);
  } else if (a.ndim == 1 && b.ndim > 1) {
    jetdl::utils::check_vecmat_shapes(a.shape, b.shape);
    return _linalg_vecmat(a, b);
  } else {
    jetdl::utils::check_matmul_shapes(a.shape, b.shape);
    return _linalg_matmul(a, b);
  }
}

jetdl::Tensor T(const jetdl::Tensor& a) { return _linalg_T(a); }

jetdl::Tensor mT(const jetdl::Tensor& a) {
  if (a.ndim < 2) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(py::str("tensor.mT only supports matrices or "
                                     "batches of matrices. Got {}D tensor.")
                                 .format(a.ndim));
  }
  return _linalg_mT(a);
}

}  // namespace linalg
}  // namespace jetdl
