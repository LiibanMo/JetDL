#pragma once

#include "tensor/python/bindings.h"
#include "tensor/tensor.h"
#include <memory>

std::unique_ptr<Tensor, TensorDeleter> linalg_dot(const Tensor &a,
                                                  const Tensor &b);

std::unique_ptr<Tensor, TensorDeleter> linalg_matmul(const Tensor &a,
                                                     const Tensor &b);

std::unique_ptr<Tensor, TensorDeleter> linalg_T(const Tensor &a);

std::unique_ptr<Tensor, TensorDeleter> linalg_mT(const Tensor &a);
