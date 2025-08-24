#include "auxiliary.h"
#include "metadata.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

size_t utils_get_count(const void* data, const void* input, const size_t N, const size_t type_size) {
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

size_t* utils_populate_lin_idxs(size_t* shape, int* strides, const size_t ndim, const size_t offset) {
    const size_t size = utils_metadata_get_size(shape, ndim);
    
    size_t* max_dim_values = (size_t*)malloc((ndim - offset) * sizeof(size_t));

    if (!max_dim_values) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int i = 0; i < ndim - offset; i++) {
        max_dim_values[i] = shape[i] - 1;
    }

    size_t* lin_idxs = (size_t*)malloc(size * sizeof(size_t));
    if (!lin_idxs) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    size_t* idx = (size_t*)malloc(ndim * sizeof(size_t));
    if (!idx) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < ndim; j++) {
            lin_idxs[i] += strides[j] * idx[j];
        }

        if (memcmp(idx, max_dim_values, ndim * sizeof(size_t)) == 0) {
            break;
        }

        for (int axis = ndim-1; axis >= 0; axis--) {
            idx[axis]++;

            if (idx[axis] <= max_dim_values[axis]) {
                break;
            }

            idx[axis] = 0;
        }
    }

    free(idx);
    idx = NULL;

    free(max_dim_values);
    max_dim_values = NULL;

    return lin_idxs;
}

size_t* utils_make_axes_positive(const int* axes, const size_t axes_ndim, const size_t ndim) {
    size_t* result_axes = (size_t*)malloc(axes_ndim * sizeof(size_t));

    if (!result_axes) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int i = 0; i < axes_ndim; i++) {
        if (axes[i] < 0) {
            result_axes[i] = axes[i] + ndim; 
        } else {
            result_axes[i] = axes[i];
        }
    }

    return result_axes;
}

void utils_erase_at_idx(void** input_ptr, const size_t idx, const size_t N, const size_t type_size) {
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