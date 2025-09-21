#include "jetdl/routines.h"

#include <memory>

#include "jetdl/routines/creation.h"
#include "jetdl/routines/manipulation.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> ones(const std::vector<size_t>& shape,
                             const bool requires_grad) {
  return _ones(shape, requires_grad);
}

std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor>& tensor,
                                const std::vector<size_t>& shape) {
  const size_t size = utils::get_size(shape);
  if (size != tensor->size) {
    throw std::runtime_error(py::str("shape '{}' invalid for reshaping.")
                                 .format(*py::tuple(py::cast(shape))));
  }
  return _reshape(tensor, shape);
}

}  // namespace jetdl
