#ifndef LINALG_PRODUCT_MATMUL_H
#define LINALG_PRODUCT_MATMUL_H

#include "jetdl/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor *c_linalg_dot(const Tensor *a, const Tensor *b);
Tensor *c_linalg_matvec(const Tensor *a, const Tensor *b);
Tensor *c_linalg_vecmat(const Tensor *a, const Tensor *b);
Tensor *c_linalg_matmul(const Tensor *a, const Tensor *b);

#ifdef __cplusplus
}
#endif

#endif
