#ifndef LINALG_LINALG_H
#define LINALG_LINALG_H

#include "tensor/tensor.h"

Tensor& linalg_dot(const Tensor& a, const Tensor& b);
Tensor& linalg_matmul(const Tensor& a, const Tensor& b);

#endif