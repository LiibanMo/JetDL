#include <vector>

#include "jetdl/routines/creation.h"
#include "jetdl/utils/metadata.h"

jetdl::Tensor _ones(const std::vector<size_t>& shape,
                    const bool requires_grad) {
  const size_t size = jetdl::utils::get_size(shape);
  auto result_data = std::make_shared<std::vector<float>>(size);

  std::fill_n(result_data->begin(), size, 1.0f);

  return jetdl::Tensor(result_data, shape, requires_grad);
}
