#include "jetdl/utils/auxiliary.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace jetdl {
namespace utils {

void fill(void* dest, const void* input, size_t N, size_t type_size) {
  char* dest_ptr = static_cast<char*>(dest);
  for (size_t i = 0; i < N; i++) {
    std::memcpy(dest_ptr + i * type_size, input, type_size);
  }
}

size_t get_count(const void* data, const void* input, size_t N,
                 size_t type_size) {
  size_t count = 0;
  const char* temp_data_ptr = static_cast<const char*>(data);
  for (size_t i = 0; i < N; i++) {
    if (std::memcmp(temp_data_ptr + i * type_size, input, type_size) == 0) {
      count++;
    }
  }
  return count;
}

std::vector<size_t> populate_linear_idxs(const std::vector<size_t>& shape,
                                         const std::vector<size_t>& strides,
                                         OpType optype) {
  size_t offset;
  if (optype == OpType::MATMUL)
    offset = 2;
  else if (optype == OpType::ARITHMETIC)
    offset = 1;
  else
    offset = 0;

  const size_t batch_ndim = shape.size() - offset;

  size_t size = 1;
  for (size_t i = 0; i < batch_ndim; i++) {
    size *= shape[i];
  }

  std::vector<size_t> lin_idxs(size, 0);
  std::vector<size_t> idx(shape.size(), 0);

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < shape.size(); j++) {
      lin_idxs[i] += strides[j] * idx[j];
    }

    for (int axis = batch_ndim - 1; axis >= 0; axis--) {
      idx[axis]++;
      if (idx[axis] < shape[axis]) {
        break;
      }
      idx[axis] = 0;
    }
  }

  return lin_idxs;
}

std::vector<size_t> make_axes_positive(const std::vector<int>& axes,
                                       size_t ndim) {
  std::vector<size_t> updated_axes;
  updated_axes.reserve(axes.size());
  for (int axis : axes) {
    if (axis < 0) {
      updated_axes.push_back(axis + ndim);
    } else {
      updated_axes.push_back(axis);
    }
  }
  return updated_axes;
}

template <typename T>
void erase_at_idx(std::vector<T>& vec, size_t idx) {
  if (idx < vec.size()) {
    vec.erase(vec.begin() + idx);
  }
}

template <typename T>
void reverse(std::vector<T>& vec, size_t start, size_t end) {
  if (start < end && end <= vec.size()) {
    std::reverse(vec.begin() + start, vec.begin() + end);
  }
}

}  // namespace utils
}  // namespace jetdl
