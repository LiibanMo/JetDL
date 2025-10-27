#ifndef JETDL_ROUTINES_CREATION_HPP
#define JETDL_ROUTINES_CREATION_HPP

#include <memory>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _copy(std::shared_ptr<Tensor>& input);

std::shared_ptr<Tensor> _zeros(const std::vector<size_t>& shape,
                               const bool requires_grad);

std::shared_ptr<Tensor> _ones(const std::vector<size_t>& shape,
                              const bool requires_grad);

std::shared_ptr<Tensor> _fill(const std::vector<size_t>& shape,
                              const float value, const bool requires_grad);

}  // namespace jetdl

#endif
