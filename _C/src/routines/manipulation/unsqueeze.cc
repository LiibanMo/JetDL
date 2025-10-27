#include <memory>
#include <vector>

#include "jetdl/routines.h"
#include "jetdl/routines/manipulation.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _unsqueeze(std::shared_ptr<Tensor>& input,
                                   const int axis) {
  int input_axis = axis;

  if (input_axis < 0) {
    input_axis += input->ndim + 1;
  }

  auto shape = std::vector<size_t>();

  size_t input_idx = 0;
  for (size_t i = 0; i < input->ndim + 1; i++) {
    if (i == input_axis) {
      shape.push_back(1);
    } else {
      shape.push_back(input->shape[input_idx]);
      input_idx++;
    }
  }

  return view(input, shape);
}

}  // namespace jetdl
