#ifndef JETDL_MATH_H
#define JETDL_MATH_H

#include "jetdl/bindings.h"
#include "jetdl/tensor.h"

#include <memory>
#include <vector>

std::unique_ptr<Tensor, TensorDeleter>
math_ops(const Tensor &a, const Tensor &b, const std::string op);

std::unique_ptr<Tensor, TensorDeleter> math_sum(const Tensor &a,
                                                std::vector<int> &axes);

#endif
