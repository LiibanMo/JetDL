#pragma once

#include "tensor/tensor.hpp"

namespace math {
    namespace ops {
        Tensor add(const Tensor& a, const Tensor& b);
        Tensor sub(const Tensor& a, const Tensor& b);
        Tensor mul(const Tensor& a, const Tensor& b);
        Tensor div(const Tensor& a, const Tensor& b);
    }
    namespace function {
        Tensor total_sum(const Tensor& tensor);
        Tensor sum_w_axes(const Tensor& tensor, const std::vector<int>& axes);
    }
}