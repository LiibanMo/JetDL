#ifndef JETDL_ROUTINES_H
#define JETDL_ROUTINES_H

#include <memory>
#include <vector>

#include "jetdl/device.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> copy(std::shared_ptr<Tensor>& input);

std::shared_ptr<Tensor> zeros(const std::vector<size_t>& shape,
                              const bool requires_grad = false,
                              const Device& device = Device::cpu());

std::shared_ptr<Tensor> ones(const std::vector<size_t>& shape,
                             const bool requires_grad = false,
                             const Device& device = Device::cpu());

std::shared_ptr<Tensor> fill(const std::vector<size_t>& shape,
                             const float scalar,
                             const bool requires_grad = false,
                             const Device& device = Device::cpu());

std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor>& tensor,
                                const std::vector<int>& shape);

std::shared_ptr<Tensor> view(std::shared_ptr<Tensor>& tensor,
                             const std::vector<size_t>& shape,
                             const bool requires_grad = false);

std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor>& input,
                                const std::vector<int>& axes = {});

std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor>& input,
                                  const int axis);

std::shared_ptr<Tensor> contiguous(std::shared_ptr<Tensor>& input);

std::string tensor_to_string(const std::shared_ptr<Tensor>& input);

}  // namespace jetdl
#endif
