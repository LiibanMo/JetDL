#ifndef JETDL_NN_H
#define JETDL_NN_H

#include <cstddef>
#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {

namespace nn {

std::shared_ptr<Tensor> linear_forward(std::shared_ptr<Tensor>& input,
                                       std::shared_ptr<Tensor>& weight,
                                       std::shared_ptr<Tensor>& bias);

std::shared_ptr<Tensor> relu_forward(std::shared_ptr<Tensor>& input);

}  // namespace nn

}  // namespace jetdl

#endif
