#ifndef JETDL_OPTIM_OPTIM_H
#define JETDL_OPTIM_OPTIM_H

#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {

namespace optim {

void sgd_step(std::shared_ptr<Tensor>& param, const float lr);

}  // namespace optim

}  // namespace jetdl

#endif