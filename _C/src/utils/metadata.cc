#include "jetdl/utils/metadata.h"

#include <vector>

namespace jetdl {
namespace utils {

size_t get_size(const std::vector<size_t>& shape) {
  const size_t ndim = shape.size();
  size_t size = 1;
  if (ndim == 0) {
    return size;
  }
  for (size_t i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

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

std::vector<size_t> get_byte_strides(const std::vector<size_t>& shape) {
  const size_t ndim = shape.size();
  if (ndim == 0) {
    return {};
  }
  auto byte_strides = std::vector<size_t>(ndim, 0);
  byte_strides[ndim - 1] = sizeof(float);
  for (int i = ndim - 2; i >= 0; i--) {
    byte_strides[i] = byte_strides[i + 1] * shape[i + 1];
  }
  return byte_strides;
}

}  // namespace utils
}  // namespace jetdl
