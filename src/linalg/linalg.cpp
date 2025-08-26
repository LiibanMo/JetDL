#include "linalg.h"
#include "product/matmul.h"
#include "python/utils/check.h"

#include <stdio.h>

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