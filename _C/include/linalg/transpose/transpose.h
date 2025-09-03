#ifndef LINALG_TRANSPOSE_H
#define LINALG_TRANSPOSE_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

    Tensor* c_linalg_T(const Tensor* a);
    Tensor* c_linalg_mT(const Tensor* a);

#ifdef __cplusplus
}
#endif

#endif