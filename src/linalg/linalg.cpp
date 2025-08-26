#include "linalg.h"
#include "linalg/transpose/transpose.h"
#include "product/matmul.h"
#include "python/utils/check.h"

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <stdio.h>

namespace py = pybind11;

Tensor& linalg_dot(const Tensor& a, const Tensor& b) {
    utils_check_dot_shapes(a.shape, a.ndim, b.shape, b.ndim);
    return *c_linalg_dot(&a, &b);
}

Tensor& linalg_matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim == 1 && b.ndim == 1) {
        utils_check_dot_shapes(a.shape, a.ndim, b.shape, b.ndim);
        return *c_linalg_dot(&a, &b);
    } else if (a.ndim > 1 && b.ndim == 1) {
        utils_check_matvec_shapes(a.shape, a.ndim, b.shape, b.ndim);
        return *c_linalg_matvec(&a, &b);
    } else if (a.ndim == 1 && b.ndim > 1) {
        utils_check_vecmat_shapes(a.shape, a.ndim, b.shape, b.ndim);
        return *c_linalg_vecmat(&a, &b);
    } else {
        utils_check_matmul_shapes(a.shape, a.ndim, b.shape, b.ndim);
        return *c_linalg_matmul(&a, &b);
    }
}

Tensor& linalg_T(const Tensor& a) {
    return *c_linalg_T(&a);
}

Tensor& linalg_mT(const Tensor& a) {
    if (a.ndim < 2) {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error(
            py::str(
                "tensor.mT only supports matrices or batches of matrices. Got {}D tensor."
            )
            .format(a.ndim)
        );
    }
    return *c_linalg_mT(&a);
}