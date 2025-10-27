#include "jetdl/linalg/transpose.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "jetdl/autograd/linalg.h"
#include "jetdl/routines.h"

namespace jetdl {

std::shared_ptr<Tensor> _linalg_T(std::shared_ptr<Tensor>& a) {
  auto shape = std::vector<size_t>(a->ndim, 0);
  auto strides = std::vector<size_t>(a->ndim, 0);

  for (size_t i = 0; i < a->ndim; i++) {
    shape[i] = a->shape[a->ndim - 1 - i];
    strides[i] = a->strides[a->ndim - 1 - i];
  }

  std::shared_ptr<Tensor> result_tensor = view(a, shape, a->requires_grad);
  result_tensor->strides = strides;

  if (std::equal(a->strides.begin(), a->strides.end(), strides.begin(),
                 strides.end())) {
    result_tensor->is_contiguous = true;
  } else {
    result_tensor->is_contiguous = false;
  }

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<TransposeBackward>(a, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _linalg_mT(std::shared_ptr<Tensor>& a) {
  auto shape = std::vector<size_t>(a->ndim, 0);
  auto strides = std::vector<size_t>(a->ndim, 0);

  for (size_t i = 0; i < a->ndim - 2; i++) {
    shape[i] = a->shape[i];
    strides[i] = a->strides[i];
  }

  shape[a->ndim - 2] = a->shape[a->ndim - 1];
  shape[a->ndim - 1] = a->shape[a->ndim - 2];

  strides[a->ndim - 2] = a->strides[a->ndim - 1];
  strides[a->ndim - 1] = a->strides[a->ndim - 2];

  std::shared_ptr<Tensor> result_tensor = view(a, shape, a->requires_grad);
  result_tensor->strides = strides;

  if (std::equal(a->strides.begin(), a->strides.end(), strides.begin(),
                 strides.end())) {
    result_tensor->is_contiguous = true;
  } else {
    result_tensor->is_contiguous = false;
  }

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<MatrixTransposeBackward>(a, result_tensor);
  }

  return result_tensor;
}

}  // namespace jetdl
