#pragma once

#include "tensor/tensor.h"
#include <memory>

std::unique_ptr<Tensor, decltype(&destroy_tensor)> linalg_dot(
    const Tensor& a, const Tensor& b
);

std::unique_ptr<Tensor, decltype(&destroy_tensor)> linalg_matmul(
    const Tensor& a, const Tensor& b
);

std::unique_ptr<Tensor, decltype(&destroy_tensor)> linalg_T(const Tensor& a);

std::unique_ptr<Tensor, decltype(&destroy_tensor)> linalg_mT(const Tensor& a);