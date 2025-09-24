#include <memory>
#include <vector>

#include "jetdl/routines/manipulation.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _make_contiguous(std::shared_ptr<Tensor>& input) {
  if (input->is_contiguous) {
    return input;
  }

  auto result_data = std::make_shared<std::vector<float>>(input->size, 0.0f);

  std::vector<size_t> current_coords(input->ndim, 0);
  for (size_t i = 0; i < input->size; ++i) {
    size_t source_offset = 0;
    for (size_t j = 0; j < input->ndim; ++j) {
      source_offset += current_coords[j] * input->strides[j];
    }

    (*result_data)[i] = (*input->_data)[source_offset];

    for (int j = input->ndim - 1; j >= 0; --j) {
      current_coords[j]++;
      if (current_coords[j] < input->shape[j]) {
        break;
      }
      current_coords[j] = 0;
    }
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->ndim);

  return result_tensor;
}

}  // namespace jetdl
