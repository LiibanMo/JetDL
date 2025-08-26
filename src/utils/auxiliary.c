#include "auxiliary.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

size_t utils_get_count(
    const void* data, const void* input, const size_t N, const size_t type_size
) {
    size_t count = 0;

    const char* temp_data_ptr = (char*)data;

    for (size_t i = 0; i < N; i++) {
        if (memcmp(temp_data_ptr + i * type_size, input, type_size) == 0) {
            count++;
        }
    }

    temp_data_ptr = NULL;

    return count;
}

size_t* utils_populate_linear_idxs(
    const size_t* shape, const size_t* strides, const size_t ndim, const OpType optype
) {
    const size_t offset = (optype == MATMUL) ? 2 : 0;
    const size_t batch_ndim = ndim - offset;

    size_t size = 1;
    for (size_t i = 0; i < batch_ndim; i++) size *= shape[i];

    size_t* lin_idxs = (size_t*)calloc(size, sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(lin_idxs, stderr, alloc_failure);

    size_t* idx = (size_t*)calloc(ndim, sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(idx, stderr, alloc_failure);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < ndim; j++) {
            lin_idxs[i] += strides[j] * idx[j];
        }

        for (int axis = batch_ndim - 1; axis >= 0; axis--) {
            idx[axis]++;
            if (idx[axis] < shape[axis]) {
                break;
            }
            idx[axis] = 0;
        }
    }

    UTILS_FREE(idx);

    return lin_idxs;

    alloc_failure:
        if (lin_idxs) UTILS_FREE(lin_idxs);
        if (idx) UTILS_FREE(idx);
        return NULL;
}

size_t* utils_make_axes_positive(
    const int* axes, const size_t axes_ndim, const size_t ndim
) {
    size_t* result_axes = (size_t*)malloc(axes_ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(result_axes, stderr, alloc_failure);

    for (int i = 0; i < axes_ndim; i++) {
        if (axes[i] < 0) result_axes[i] = axes[i] + ndim; 
        else result_axes[i] = axes[i];
    }

    return result_axes;

    alloc_failure:
        if(result_axes) UTILS_FREE(result_axes);
        return NULL;
}

void utils_erase_at_idx(
    void** input_ptr, const size_t idx, const size_t N, const size_t type_size
) {
    if (idx >= N) {
        fprintf(stderr, "erase: idx out of bounds.\n");
        return;
    }

    if (idx < N - 1) {
        memmove(
            (char*)*input_ptr + idx * type_size, 
            (char*)*input_ptr + (idx + 1) * type_size, 
            (N - 1 - idx) * type_size
        );
    }
    
    void* temp_ptr = (void*)realloc(*input_ptr, (N-1) * type_size);
    if (N - 1 > 0 && !temp_ptr) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }

    *input_ptr = temp_ptr;
    temp_ptr = NULL;
}
