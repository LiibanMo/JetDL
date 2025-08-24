#include "metadata.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

size_t utils_metadata_get_size(const size_t* shape, const size_t ndim) {
    if (ndim == 0) {
        return 1;
    }

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) size *= shape[i];
    return size;
}

size_t* utils_metadata_get_strides(const size_t* shape, const size_t ndim) {
    if (ndim == 0) {
        return NULL;
    }

    size_t* strides = (size_t*)malloc(ndim * sizeof(ndim));
    if (!strides) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    strides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; --i) strides[i] = strides[i+1] * shape[i+1];

    return strides;
}

void utils_metadata_assign_basics(Tensor* mutable_tensor, const size_t* shape, const size_t ndim) {
    mutable_tensor->shape = NULL;
    if (ndim > 0) {
        mutable_tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
        if (!mutable_tensor->shape && ndim > 0) {
            fprintf(stderr, "Memory allocation failed.\n");
            return;
        } 
        memcpy(mutable_tensor->shape, shape, ndim * sizeof(size_t));
    } else {
        mutable_tensor->shape = NULL;
    }

    mutable_tensor->ndim = ndim;
    mutable_tensor->size = utils_metadata_get_size(shape, ndim);
    mutable_tensor->strides = utils_metadata_get_strides(shape, ndim);
    mutable_tensor->is_contiguous = true;
}