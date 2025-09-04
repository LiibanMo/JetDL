#ifndef JETDL_LINALG_H
#define JETDL_LINALG_H

#include "jetdl/bindings.h"
#include "jetdl/tensor.h"

std::unique_ptr<Tensor, TensorDeleter> linalg_dot(const Tensor &a,
                                                  const Tensor &b);

std::unique_ptr<Tensor, TensorDeleter> linalg_matmul(const Tensor &a,
                                                     const Tensor &b);

std::unique_ptr<Tensor, TensorDeleter> linalg_T(const Tensor &a);

std::unique_ptr<Tensor, TensorDeleter> linalg_mT(const Tensor &a);

#endif
