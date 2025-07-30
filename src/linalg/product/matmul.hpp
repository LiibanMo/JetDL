#pragma once

#include "tensor/tensor.hpp"

Tensor c_dot(const Tensor& a, const Tensor& b);
Tensor c_matvec(const Tensor& a, const Tensor& b);
Tensor c_vecmat(const Tensor& a, const Tensor& b);
Tensor c_matmul(const Tensor& a, const Tensor& b);