#pragma once

#include "tensor/tensor.hpp"

namespace math {
    namespace ops {
        Tensor add(Tensor& a, Tensor& b);
        Tensor sub(Tensor& a, Tensor& b);
        Tensor mul(Tensor& a, Tensor& b);
        Tensor div(Tensor& a, Tensor& b);
    }
    namespace function {
        Tensor total_sum(const Tensor& tensor);
        Tensor sum_w_axes(const Tensor& tensor, const std::vector<int>& axes);
    }
}