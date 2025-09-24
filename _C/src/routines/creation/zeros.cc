#include <memory>
#include <vector>

#include "jetdl/routines/creation.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> _zeros(const std::vector<size_t>& shape,
                               const bool requires_grad) {
  const size_t size = utils::get_size(shape);
  auto result_data = std::make_shared<std::vector<float>>(size, 0.0f);
  auto result_tensor =
      std::make_shared<Tensor>(result_data, shape, requires_grad);

  return result_tensor;
}

}  // namespace jetdl
