#pragma once

#include "tensor/tensor.hpp"

namespace creation {
    Tensor ones(const std::vector<int>& shape, const bool requires_grad);
}