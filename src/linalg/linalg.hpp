#pragma once

#include "../tensor/tensor.hpp"

namespace linalg {
    Tensor dot(const Tensor& a, const Tensor& b);
    Tensor matmul(const Tensor& a, const Tensor& b);
}