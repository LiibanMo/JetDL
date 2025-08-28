#include "auxiliary.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void utils_fill(void* dest, const void* input, const size_t N, const size_t type_size) {
    char* dest_ptr = (char*)dest;
    for (size_t i = 0; i < N; i++) memcpy(dest_ptr + i * type_size, input, type_size);
}

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

size_t* utils_make_axes_positive(const int* axes, const size_t naxes, const size_t ndim) {
    size_t* updated_axes = (size_t*)malloc(naxes * sizeof(size_t));
    if (!updated_axes) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (size_t i = 0; i < naxes; i++) {
        if (axes[i] < 0) updated_axes[i] = axes[i] + ndim; 
        else updated_axes[i] = axes[i];
    }
    return updated_axes;
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

void utils_reverse(void** input_ptr, const size_t START, const size_t END, const size_t type_size) {
    if (END < START) {
        fprintf(stderr, "END < START\n");
        return;
    } else if (END == START) {
        return;
    }

    char* start_ptr = (char*)*input_ptr + START * type_size;
    char* end_ptr = (char*)*input_ptr + (END - 1) * type_size;

    char temp_value[type_size];
    while (start_ptr < end_ptr) {
        memcpy(temp_value, start_ptr, type_size);
        memcpy(start_ptr, end_ptr, type_size);
        memcpy(end_ptr, temp_value, type_size);

        start_ptr += type_size;
        end_ptr -= type_size;
    }
}

int utils_compare_ints(const void* ptrA, const void* ptrB) {
    int a = *(int*)ptrA;
    int b = *(int*)ptrB;

    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}

int utils_compare_size_t(const void* ptrA, const void* ptrB) {
    size_t a = *(size_t*)ptrA;
    size_t b = *(size_t*)ptrB;

    if (a > b) return 1;
    if (a < b) return -1;
    return 0; 
}

int utils_compare_float(const void* ptrA, const void* ptrB) {
    float a = *(float*)ptrA;
    float b = *(float*)ptrB;

    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}