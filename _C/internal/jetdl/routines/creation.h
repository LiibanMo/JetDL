#ifndef JETDL_ROUTINES_CREATION_HPP
#define JETDL_ROUTINES_CREATION_HPP

#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _ones(const std::vector<size_t>& shape,
                              const bool requires_grad);

}

#endif
