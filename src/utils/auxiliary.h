#pragma once

#include <stddef.h>

#define UTILS_GET_MAX(a, b) ((a > b) ? a : b)

size_t utils_get_count(const void* data, const void* input, const size_t N, const size_t type_size);

#define UTILS_NXT_MULTIPLE(input, factor) (((input + factor - 1) / factor) * factor)

size_t* utils_populate_lin_idxs(size_t* shape, int* strides, const size_t ndim, const size_t offset);

size_t* utils_make_axes_positive(const int* axes, const size_t axes_ndim, const size_t ndim);

void utils_erase_at_idx(void** input_ptr, const size_t idx, const size_t N, const size_t type_size);