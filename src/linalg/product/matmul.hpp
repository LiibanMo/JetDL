#pragma once

#include "../../tensor/tensor.hpp"

Tensor c_matvec(const Tensor& a, const Tensor& b);
Tensor c_matmul_batched(const Tensor& a, const Tensor& b);