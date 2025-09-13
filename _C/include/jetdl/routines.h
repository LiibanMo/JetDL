#ifndef JETDL_ROUTINES_H
#define JETDL_ROUTINES_H

#include "jetdl/tensor.h"

namespace jetdl {

jetdl::Tensor ones(const std::vector<size_t>& shape, const bool requires_grad);

}  // namespace jetdl
#endif
