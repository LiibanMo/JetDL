#include "metadata.h"
#include "utils/auxiliary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t utils_metadata_get_size(const size_t *shape, const size_t ndim) {
  size_t size = 1;

  if (ndim == 0) {
    return size;
  }

  for (size_t i = 0; i < ndim; ++i)
    size *= shape[i];
  return size;
}

size_t *utils_metadata_get_strides(const size_t *shape, const size_t ndim) {
  size_t *strides = (size_t *)calloc(ndim, sizeof(size_t));
  UTILS_CHECK_ALLOC_FAILURE(strides, stderr, alloc_failure);

  if (ndim == 0) {
    return strides;
  }

  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];

  return strides;

alloc_failure:
  if (strides)
    UTILS_FREE(strides);
  return NULL;
}

void utils_metadata_assign_basics(Tensor *mutable_tensor, size_t *shape,
                                  const size_t ndim) {
  if (ndim > 0 && shape) {
    mutable_tensor->shape = shape;
    shape = NULL;
  } else if (ndim == 0) {
    mutable_tensor->shape = (size_t *)calloc(0, sizeof(size_t));
  } else {
    fprintf(stderr, "ndim cannot be zero and shape not NULL\n");
    exit(1);
  }

  mutable_tensor->ndim = ndim;
  mutable_tensor->size = utils_metadata_get_size(mutable_tensor->shape, ndim);
  mutable_tensor->strides =
      utils_metadata_get_strides(mutable_tensor->shape, ndim);
  mutable_tensor->is_contiguous = true;
}
