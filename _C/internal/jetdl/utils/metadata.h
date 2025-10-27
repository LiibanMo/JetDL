#ifndef JETDL_UTILS_METADATA_HPP
#define JETDL_UTILS_METADATA_HPP

#include <vector>

namespace jetdl {
namespace utils {

template <typename T>
T get_size(const std::vector<T>& shape) {
  const size_t ndim = shape.size();
  T size = 1;
  if (ndim == 0) {
    return size;
  }
  for (size_t i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

std::vector<size_t> get_strides(const std::vector<size_t>& shape);

}  // namespace utils
}  // namespace jetdl

#endif  // JETDL_UTILS_METADATA_HPP
