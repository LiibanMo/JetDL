#pragma once

#include "tensor/tensor.hpp"

Tensor c_total_sum(const Tensor& tensor);
Tensor c_sum_w_axes(const Tensor& tensor, const std::vector<int>& axes);
Tensor c_sum_to_size(const Tensor& tensor, const std::vector<int>& shape);