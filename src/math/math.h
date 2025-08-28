#pragma once

#include "tensor/tensor.h"

#include <vector>

Tensor& math_ops(const Tensor& a, const Tensor& b, const std::string op);
Tensor& math_sum(const Tensor& a, std::vector<int>& axes);