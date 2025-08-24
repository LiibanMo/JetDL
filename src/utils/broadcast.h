#pragma once

#include <stddef.h>

size_t** utils_broadcast_get_strides(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB, const bool matmul);
size_t* utils_broadcast_get_result_shape(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB, const bool matmul);
size_t utils_broadcast_get_batch_size(const size_t* shape, const size_t ndim);