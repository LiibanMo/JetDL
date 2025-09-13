#include "jetdl/routines.h"

#include "jetdl/routines/creation.h"

namespace jetdl {

Tensor ones(const std::vector<size_t>& shape, const bool requires_grad) {
  return _ones(shape, requires_grad);
}

}  // namespace jetdl
