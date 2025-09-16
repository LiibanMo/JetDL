#include "jetdl/linalg/transpose.h"

#include <cstring>
#include <vector>

jetdl::Tensor _linalg_T(const jetdl::Tensor& a) {
  auto shape = std::vector<size_t>(a.ndim, 0);
  auto strides = std::vector<size_t>(a.ndim, 0);

  for (size_t i = 0; i < a.ndim; i++) {
    shape[i] = a.shape[a.ndim - 1 - i];
    strides[i] = a.strides[a.ndim - 1 - i];
  }

  jetdl::Tensor result_tensor = a.view(shape);
  result_tensor.strides = strides;

  if (std::equal(a.strides.begin(), a.strides.end(), strides.begin(),
                 strides.end())) {
    result_tensor.is_contiguous = true;
  } else {
    result_tensor.is_contiguous = false;
  }

  return result_tensor;
}

jetdl::Tensor _linalg_mT(const jetdl::Tensor& a) {
  auto shape = std::vector<size_t>(a.ndim, 0);
  auto strides = std::vector<size_t>(a.ndim, 0);

  for (size_t i = 0; i < a.ndim - 2; i++) {
    shape[i] = a.shape[i];
    strides[i] = a.strides[i];
  }

  shape[a.ndim - 2] = a.shape[a.ndim - 1];
  shape[a.ndim - 1] = a.shape[a.ndim - 2];

  strides[a.ndim - 2] = a.strides[a.ndim - 1];
  strides[a.ndim - 1] = a.strides[a.ndim - 2];

  jetdl::Tensor result_tensor = a.view(shape);
  result_tensor.strides = strides;

  if (std::equal(a.strides.begin(), a.strides.end(), strides.begin(),
                 strides.end())) {
    result_tensor.is_contiguous = true;
  } else {
    result_tensor.is_contiguous = false;
  }

  return result_tensor;
}
