#ifndef JETDL_ROUTINES_MANIPULATION_H
#define JETDL_ROUTINES_MANIPULATION_H

#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _reshape(const Tensor& tensor,
                                 const std::vector<size_t>& shape);

}

#endif
