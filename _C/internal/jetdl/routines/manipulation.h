#ifndef JETDL_ROUTINES_MANIPULATION_H
#define JETDL_ROUTINES_MANIPULATION_H

#include <memory>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _reshape(std::shared_ptr<Tensor>& tensor,
                                 const std::vector<size_t>& shape);

std::shared_ptr<Tensor> _squeeze(std::shared_ptr<Tensor>& input,
                                 const std::vector<int>& axes);

std::shared_ptr<Tensor> _unsqueeze(std::shared_ptr<Tensor>& input,
                                   const int axis);

std::shared_ptr<Tensor> _make_contiguous(std::shared_ptr<Tensor>& input);
}  // namespace jetdl

#endif
