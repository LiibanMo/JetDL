#pragma once

#include "tensor/tensor.h"

Tensor* c_linalg_dot(const Tensor* a, const Tensor* b);
Tensor* c_linalg_matvec(const Tensor* a, const Tensor* b);
Tensor* c_linalg_vecmat(const Tensor* a, const Tensor* b);
Tensor* c_linalg_matmul(const Tensor* a, const Tensor* b);