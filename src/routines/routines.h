#pragma once

#include "tensor/python/bindings.h"
#include "tensor/tensor.h"

#include <memory>
#include <vector>

std::unique_ptr<Tensor, TensorDeleter>
routines_ones(const std::vector<size_t> shape);
