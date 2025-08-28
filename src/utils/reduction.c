#include "reduction.h"
#include "utils/auxiliary.h"

#include <stdlib.h>
#include <stdio.h>

size_t* utils_reduction_get_shape(
    const size_t* shape, const size_t ndim, const size_t* axes, const size_t naxes
) {
    const size_t result_ndim = ndim - naxes;
    size_t* result_shape = (size_t*)malloc(result_ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(result_shape, stderr, alloc_failure);
    
    size_t* result_shape_ptr = result_shape;
    const size_t* axes_ptr = axes;

    for (size_t i = 0; i < ndim; i++) {
        if (i == *axes_ptr) {
            axes_ptr++;
        } else {
            *result_shape_ptr = shape[i];
            result_shape_ptr++;
        }
    }

    return result_shape;

    alloc_failure:
        if (result_shape) UTILS_FREE(result_shape);
        return NULL;
}

size_t* utils_reduction_get_dest_strides(
    const size_t* original_shape, const size_t original_ndim, const size_t* result_strides,
    const size_t* axes, const size_t naxes
) {
    size_t* reduction_strides = (size_t*)malloc(original_ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(reduction_strides, stderr, alloc_failure);

    size_t axes_idx = 0;
    size_t result_strides_idx = 0;

    for (size_t i = 0; i < original_ndim; i++) {
        if (i == axes[axes_idx]) {
            reduction_strides[i] = 0;
            axes_idx++;
        } else {
            reduction_strides[i] = result_strides[result_strides_idx];
            result_strides_idx++;
        }
    }

    return reduction_strides;

    alloc_failure:
        if (reduction_strides) UTILS_FREE(reduction_strides);
        return NULL;
}
