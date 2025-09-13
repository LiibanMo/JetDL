#include "jetdl/utils/reduction.h"

#include <vector>

namespace jetdl {
namespace utils {

std::vector<size_t> get_shape(const std::vector<size_t>& shape,
                              const std::vector<size_t>& axes) {
  std::vector<size_t> result_shape;
  result_shape.reserve(shape.size() - axes.size());

  size_t axes_idx = 0;
  for (size_t i = 0; i < shape.size(); i++) {
    if (axes_idx < axes.size() && i == axes[axes_idx]) {
      axes_idx++;
    } else {
      result_shape.push_back(shape[i]);
    }
  }
  return result_shape;
}

std::vector<size_t> get_dest_strides(const std::vector<size_t>& original_shape,
                                     const std::vector<size_t>& result_strides,
                                     const std::vector<size_t>& axes) {
  std::vector<size_t> reduction_strides(original_shape.size());

  size_t axes_idx = 0;
  size_t result_strides_idx = 0;

  for (size_t i = 0; i < original_shape.size(); i++) {
    if (axes_idx < axes.size() && i == axes[axes_idx]) {
      reduction_strides[i] = 0;
      axes_idx++;
    } else {
      reduction_strides[i] = result_strides[result_strides_idx];
      result_strides_idx++;
    }
  }

  return reduction_strides;
}

}  // namespace utils
}  // namespace jetdl
