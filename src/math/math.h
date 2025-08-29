#pragma once

#include "tensor/tensor.h"

#include <memory.h>
#include <vector>

std::unique_ptr<Tensor, decltype(&destroy_tensor)> math_ops(
    const Tensor& a, const Tensor& b, const std::string op
);
std::unique_ptr<Tensor, decltype(&destroy_tensor)> math_sum(
    const Tensor& a, std::vector<int>& axes
);