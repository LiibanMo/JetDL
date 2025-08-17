#pragma once

#include "tensor/tensor.hpp"

Tensor c_ops(Tensor& a, Tensor& b, const std::string op);
Tensor c_ops_scalar_a(Tensor& a, Tensor& b, const std::string op);
Tensor c_ops_scalar_b(Tensor& a, Tensor& b, const std::string op);
Tensor c_ops_scalars(Tensor& a, Tensor& b, const std::string op);