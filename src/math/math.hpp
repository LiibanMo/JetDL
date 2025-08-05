#pragma once

#include "tensor/tensor.hpp"

namespace math {

    Tensor add(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, const Tensor& b);

}