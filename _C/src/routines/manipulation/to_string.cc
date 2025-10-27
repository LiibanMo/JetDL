#include <sstream>
#include <string>
#include <vector>

#include "jetdl/routines/manipulation.h"
#include "jetdl/tensor.h"

namespace jetdl {

namespace {

void to_string_recursive(std::stringstream& ss,
                         const std::shared_ptr<Tensor>& tensor, size_t dim,
                         std::vector<size_t>& indices) {
  ss << "[";
  if (dim == tensor->ndim - 1) {
    for (size_t i = 0; i < tensor->shape[dim]; ++i) {
      indices[dim] = i;
      size_t offset = 0;
      for (size_t d = 0; d < tensor->ndim; ++d) {
        offset += indices[d] * tensor->strides[d];
      }
      ss << tensor->_data[offset];
      if (i < tensor->shape[dim] - 1) {
        ss << ", ";
      }
    }
  } else {
    for (size_t i = 0; i < tensor->shape[dim]; ++i) {
      indices[dim] = i;
      to_string_recursive(ss, tensor, dim + 1, indices);
      if (i < tensor->shape[dim] - 1) {
        ss << ",\n";
        ss << std::string(dim + 1, ' ');
      }
    }
  }
  ss << "]";
}

}  // namespace

std::string _tensor_to_string(const std::shared_ptr<Tensor>& input) {
  if (input->ndim == 0) {
    return std::to_string(input->_data[0]);
  }

  std::stringstream ss;
  std::vector<size_t> indices(input->ndim, 0);
  to_string_recursive(ss, input, 0, indices);
  return ss.str();
}

}  // namespace jetdl
