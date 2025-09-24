#include "jetdl/routines.h"

#include <memory>
#include <vector>

#include "jetdl/routines/creation.h"
#include "jetdl/routines/manipulation.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/check.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> zeros(const std::vector<size_t>& shape,
                              const bool requires_grad) {
  return _zeros(shape, requires_grad);
}

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

std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor>& input,
                                const std::vector<int>& axes) {
  utils::check_axes(input->shape, axes);
  return _squeeze(input, axes);
}

std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor>& input,
                                  const int axes) {
  utils::check_axes(input->shape, std::vector<int>{axes},
                    utils::SubModule::ROUTINES);
  return _unsqueeze(input, axes);
}

std::shared_ptr<Tensor> contiguous(std::shared_ptr<Tensor>& input) {
  return _make_contiguous(input);
}

}  // namespace jetdl
