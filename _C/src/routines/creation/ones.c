#include "jetdl/C/routines/creation.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/metadata.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *c_routines_ones(const size_t *input_shape, const size_t ndim) {
  size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
  UTILS_CHECK_ALLOC_FAILURE(shape, stderr, alloc_failure);
  memcpy(shape, input_shape, ndim * sizeof(size_t));

  const size_t _DATA_SIZE = utils_metadata_get_size(shape, ndim);
  float *_data = (float *)malloc(_DATA_SIZE * sizeof(float));
  UTILS_CHECK_ALLOC_FAILURE(_data, stderr, alloc_failure);

  for (size_t i = 0; i < _DATA_SIZE; i++)
    _data[i] = 1.0;

  return create_tensor(_data, shape, ndim);

alloc_failure:
  if (shape)
    UTILS_FREE(shape);
  if (_data)
    UTILS_FREE(_data);
  return NULL;
}
