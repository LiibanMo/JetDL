#include "jetdl/utils/auxiliary.h"

#include <cstring>
#include <vector>

namespace jetdl {
namespace utils {

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

  auto lin_idxs = std::vector<size_t>(size, 0);
  auto idx = std::vector<size_t>(shape.size(), 0);

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

}  // namespace utils
}  // namespace jetdl
