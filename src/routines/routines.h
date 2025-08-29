#pragma once

#include "tensor/tensor.h"

#include <memory>
#include <vector>

std::unique_ptr<Tensor, decltype(&destroy_tensor)> routines_ones(
    const std::vector<size_t> shape
);