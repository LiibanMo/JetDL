#pragma once

#include "tensor/tensor.h"

Tensor& linalg_dot(const Tensor& a, const Tensor& b);
Tensor& linalg_matmul(const Tensor& a, const Tensor& b);
Tensor& linalg_T(const Tensor& a);
Tensor& linalg_mT(const Tensor& a);