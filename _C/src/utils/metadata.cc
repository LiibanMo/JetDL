#include "jetdl/utils/metadata.h"

#include <vector>

namespace jetdl {
namespace utils {

std::vector<size_t> get_strides(const std::vector<size_t>& shape) {
  const size_t ndim = shape.size();
  if (ndim == 0) {
    return {};
  }
  auto strides = std::vector<size_t>(ndim, 1);
  for (int i = ndim - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

}  // namespace utils
}  // namespace jetdl
