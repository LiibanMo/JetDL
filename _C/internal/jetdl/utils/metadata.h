#ifndef UTILS_METADATA_H
#define UTILS_METADATA_H

#include "jetdl/tensor.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t utils_metadata_get_size(const size_t *shape, const size_t ndim);
size_t *utils_metadata_get_strides(const size_t *shape, const size_t ndim);
void utils_metadata_assign_basics(Tensor *mutable_tensor, size_t *shape,
                                  const size_t ndim);
void utils_metadata_assign_grad(Tensor *mutable_tensor, Tensor **prev_tensors,
                                size_t nprev_tensors);

#ifdef __cplusplus
}
#endif

#endif
