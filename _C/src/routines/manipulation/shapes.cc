#include <memory>

#include "jetdl/routines/manipulation.h"

namespace jetdl {

std::shared_ptr<Tensor> _reshape(const Tensor& tensor,
                                 const std::vector<size_t>& shape) {
  return tensor.view(shape);
}

}  // namespace jetdl
