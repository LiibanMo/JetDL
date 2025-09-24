#include <algorithm>
#include <memory>
#include <vector>

#include "jetdl/routines/manipulation.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _squeeze(std::shared_ptr<Tensor>& input,
                                 const std::vector<int>& axes) {
  auto shape = std::vector<size_t>();

  if (axes.empty()) {
    for (const auto& dim : input->shape) {
      if (dim != 1) {
        shape.push_back(dim);
      }
    }
  } else {
    for (size_t i = 0; i < input->ndim; i++) {
      auto it = std::find(axes.begin(), axes.end(), i);
      const size_t dim = input->shape[i];
      if (dim != 1 || it == axes.end()) {
        shape.push_back(dim);
      }
    }
  }

  return input->view(shape);
}

}  // namespace jetdl
