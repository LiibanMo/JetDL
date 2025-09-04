#ifndef UTILS_BROADCAST_H
#define UTILS_BROADCAST_H

#include "jetdl/utils/auxiliary.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t **utils_broadcast_get_strides(const size_t *shapeA, const size_t ndimA,
                                     const size_t *shapeB, const size_t ndimB,
                                     const OpType optype);
size_t *utils_broadcast_get_result_shape(const size_t *shapeA,
                                         const size_t ndimA,
                                         const size_t *shapeB,
                                         const size_t ndimB,
                                         const OpType optype);
size_t utils_broadcast_get_batch_size(const size_t *shape, const size_t ndim);

#if __cplusplus
}
#endif

#endif
