#include "jetdl/linalg.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>

#include "jetdl/linalg/product.h"
#include "jetdl/linalg/transpose.h"
#include "jetdl/routines.h"
#include "jetdl/utils/check.h"

namespace py = pybind11;

namespace jetdl {
namespace linalg {

std::shared_ptr<Tensor> dot(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b) {
  utils::check_dot_shapes(a->shape, b->shape);
  return _linalg_dot(a, b);
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor>& a,
                               std::shared_ptr<Tensor>& b) {
  if (a->ndim == 1 && b->ndim == 1) {
    utils::check_dot_shapes(a->shape, b->shape);
    return _linalg_dot(a, b);
  } else if (a->ndim > 1 && b->ndim == 1) {
    utils::check_matvec_shapes(a->shape, b->shape);
    std::shared_ptr<Tensor> operandA = contiguous(a);
    return _linalg_matvec(operandA, b);
  } else if (a->ndim == 1 && b->ndim > 1) {
    utils::check_vecmat_shapes(a->shape, b->shape);
    std::shared_ptr<Tensor> operandB = contiguous(b);
    return _linalg_vecmat(a, operandB);
  } else {
    utils::check_matmul_shapes(a->shape, b->shape);
    std::shared_ptr<Tensor> operandA = contiguous(a);
    std::shared_ptr<Tensor> operandB = contiguous(b);
    return _linalg_matmul(operandA, operandB);
  }
}

std::shared_ptr<Tensor> T(std::shared_ptr<Tensor>& a) { return _linalg_T(a); }

std::shared_ptr<Tensor> mT(std::shared_ptr<Tensor>& a) {
  if (a->ndim < 2) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(py::str("tensor.mT only supports matrices or "
                                     "batches of matrices. Got {}D tensor.")
                                 .format(a->ndim));
  }
  return _linalg_mT(a);
}

}  // namespace linalg
}  // namespace jetdl
