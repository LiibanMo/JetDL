#include "math/functions/reduction.h"
#include "math/functions/kernel.h"
#include "tensor/tensor.h"
#include "utils/auxiliary.h"
#include "utils/metadata.h"
#include "utils/reduction.h"

#include <stdio.h>
#include <stdlib.h>

Tensor *c_math_total_sum(const Tensor *a) {
  float *result_data = (float *)calloc(1, sizeof(float));
  if (!result_data) {
    fprintf(stderr, "Memory allocation failed.\n");
    return NULL;
  }
  c_total_sum_cpu(result_data, a->_data, a->size);

  return create_tensor(result_data, NULL, 0);
}

Tensor *c_math_sum_over_axes(const Tensor *a, const int *axes,
                             const size_t naxes) {
  size_t *updated_axes = utils_make_axes_positive(axes, naxes, a->ndim);
  qsort(updated_axes, naxes, sizeof(size_t), utils_compare_size_t);

  const size_t ndim = a->ndim - naxes;

  size_t *shape =
      utils_reduction_get_shape(a->shape, a->ndim, updated_axes, naxes);

  const size_t RESULT_DATA_SIZE = utils_metadata_get_size(shape, ndim);
  float *result_data = (float *)calloc(RESULT_DATA_SIZE, sizeof(float));
  if (!result_data) {
    fprintf(stderr, "Memory allocation failed.\n");
    return NULL;
  }

  size_t *result_strides = utils_metadata_get_strides(shape, ndim);

  size_t *dest_strides = utils_reduction_get_dest_strides(
      a->shape, a->ndim, result_strides, updated_axes, naxes);

  UTILS_FREE(updated_axes);
  UTILS_FREE(result_strides);

  size_t *dest_idxs =
      utils_populate_linear_idxs(a->shape, dest_strides, a->ndim, REDUCTION);

  UTILS_FREE(dest_strides);

  c_sum_over_axes_cpu(result_data, a->_data, dest_idxs, a->size);

  UTILS_FREE(dest_idxs);

  return create_tensor(result_data, shape, ndim);
}
