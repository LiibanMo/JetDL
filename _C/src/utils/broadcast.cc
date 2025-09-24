#include "jetdl/utils/broadcast.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "jetdl/utils/metadata.h"

namespace jetdl {
namespace utils {

std::pair<std::vector<size_t>, std::vector<size_t>> get_strides(
    const std::vector<size_t>& shapeA, const std::vector<size_t>& shapeB,
    OpType optype) {
  if (optype == OpType::DOT) {
    return {};
  }

  const size_t max_ndim = std::max(shapeA.size(), shapeB.size());

  auto broadcasted_stridesA = std::vector<size_t>(max_ndim, 0);
  auto broadcasted_stridesB = std::vector<size_t>(max_ndim, 0);

  const std::vector<size_t>& stridesA = get_strides(shapeA);
  const std::vector<size_t>& stridesB = get_strides(shapeB);

  const size_t offset = (optype == OpType::MATMUL) ? 2 : 0;

  for (int i = max_ndim - offset - 1; i >= 0; i--) {
    const int idxA = i - max_ndim + shapeA.size();
    const int idxB = i - max_ndim + shapeB.size();

    const size_t dimA = (idxA < 0) ? 1 : shapeA[idxA];
    const size_t dimB = (idxB < 0) ? 1 : shapeB[idxB];

    broadcasted_stridesA[i] = (dimA == 1 && dimA < dimB) ? 0 : stridesA[idxA];
    broadcasted_stridesB[i] = (dimB == 1 && dimB < dimA) ? 0 : stridesB[idxB];
  }

  return {broadcasted_stridesA, broadcasted_stridesB};
}

std::vector<size_t> get_result_shape(const std::vector<size_t>& shapeA,
                                     const std::vector<size_t>& shapeB,
                                     OpType optype) {
  if (optype == OpType::DOT) {
    return {};
  }

  const int max_ndim = std::max(shapeA.size(), shapeB.size());

  auto result_shape = std::vector<size_t>(max_ndim, 0);

  int offset = 0;
  if (optype == OpType::MATMUL) {
    result_shape[max_ndim - 2] = shapeA[shapeA.size() - 2];
    result_shape[max_ndim - 1] = shapeB[shapeB.size() - 1];
    offset = 2;
  }

  for (int i = max_ndim - offset - 1; i >= 0; i--) {
    const int idxA = i - max_ndim + shapeA.size();
    const int idxB = i - max_ndim + shapeB.size();

    const int dimA = (idxA < 0) ? 1 : shapeA[idxA];
    const int dimB = (idxB < 0) ? 1 : shapeB[idxB];

    result_shape[i] = std::max(dimA, dimB);
  }

  if (optype == OpType::MATMUL) {
    if (shapeA.size() == 1) {
      result_shape.erase(result_shape.begin() + max_ndim - 2);
    } else if (shapeB.size() == 1) {
      result_shape.erase(result_shape.begin() + max_ndim - 1);
    }
  }

  return result_shape;
}

size_t get_batch_size(const std::vector<size_t>& shape) {
  size_t batch_size = 1;
  if (shape.size() <= 2) {
    return batch_size;
  }
  for (size_t i = 0; i < shape.size() - 2; ++i) {
    batch_size *= shape[i];
  }
  return batch_size;
}

std::vector<size_t> get_broadcasted_axes(
    const std::vector<size_t>& original_shape,
    const std::vector<size_t>& broadcasted_shape) {
  const size_t broadcasted_ndim = broadcasted_shape.size();
  const size_t original_ndim = original_shape.size();

  if (broadcasted_ndim < original_ndim) {
    throw std::runtime_error(
        "INTERNAL: broadcasted_ndim cannot be smaller than original_ndim.\n");
  }

  auto broadcasted_axes = std::vector<size_t>();

  for (size_t i = 0; i < broadcasted_ndim; ++i) {
    long original_i = (long)i - ((long)broadcasted_ndim - (long)original_ndim);

    if (original_i < 0) {
      broadcasted_axes.push_back(i);
    } else {
      if (original_shape[original_i] == 1 && broadcasted_shape[i] > 1) {
        broadcasted_axes.push_back(i);
      }
    }
  }
  return broadcasted_axes;
}

}  // namespace utils
}  // namespace jetdl
