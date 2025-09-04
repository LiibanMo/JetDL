#ifndef JETDL_ROUTINES_H
#define JETDL_ROUTINES_H

#include "jetdl/bindings.h"
#include "jetdl/tensor.h"

std::unique_ptr<Tensor, TensorDeleter>
routines_ones(const std::vector<size_t> shape);

#endif
