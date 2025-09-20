#ifndef JETDL_ROUTINES_H
#define JETDL_ROUTINES_H

#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {

Tensor ones(const std::vector<size_t>& shape, const bool requires_grad);
std::shared_ptr<Tensor> reshape(const Tensor& tensor,
                                const std::vector<size_t>& shape);

}  // namespace jetdl
#endif
