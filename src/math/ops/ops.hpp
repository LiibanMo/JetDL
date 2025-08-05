#pragma once

#include "tensor/tensor.hpp"

Tensor c_ops(const Tensor& a, const Tensor& b, const std::string op);
Tensor c_ops_scalar_a(const float a, const Tensor& b, const std::string op);
Tensor c_ops_scalar_b(const Tensor& a, const float b, const std::string op);
Tensor c_ops_scalars(const Tensor& a, const Tensor& b, const std::string op);