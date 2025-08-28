#include "broadcast.h"
#include "auxiliary.h"
#include "metadata.h"

#include <stdlib.h>
#include <stdio.h>

size_t** utils_broadcast_get_strides(
    const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB, const OpType optype
) {
    if (optype == DOT) return NULL;
    
    const size_t max_ndim = UTILS_GET_MAX(ndimA, ndimB);

    size_t* broadcasted_stridesA = (size_t*)calloc(max_ndim, sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(broadcasted_stridesA, stderr, alloc_failure);

    size_t* broadcasted_stridesB = (size_t*)calloc(max_ndim, sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(broadcasted_stridesB, stderr, alloc_failure);

    size_t** broadcasted_strides_ptrs = (size_t**)malloc(2 * sizeof(size_t*));
    UTILS_CHECK_ALLOC_FAILURE(broadcasted_strides_ptrs, stderr, alloc_failure);

    broadcasted_strides_ptrs[0] = broadcasted_stridesA;
    broadcasted_strides_ptrs[1] = broadcasted_stridesB;

    size_t* stridesA = utils_metadata_get_strides(shapeA, ndimA);
    size_t* stridesB = utils_metadata_get_strides(shapeB, ndimB);

    const size_t offset = (optype == MATMUL) ? 2 : 0;

    for (int i = max_ndim - offset - 1; i >= 0; i--) {
        const int idxA = i - max_ndim + ndimA;
        const int idxB = i - max_ndim + ndimB;

        const size_t dimA = (idxA < 0) ? 1 : shapeA[idxA];
        const size_t dimB = (idxB < 0) ? 1 : shapeB[idxB];

        broadcasted_stridesA[i] = (dimA == 1 && dimA < dimB) ? 0 : stridesA[idxA];
        broadcasted_stridesB[i] = (dimB == 1 && dimB < dimA) ? 0 : stridesB[idxB];
    }

    UTILS_FREE(stridesA);
    UTILS_FREE(stridesB);

    return broadcasted_strides_ptrs;

    alloc_failure:
        if (broadcasted_stridesA) UTILS_FREE(broadcasted_stridesA);
        if (broadcasted_stridesB) UTILS_FREE(broadcasted_stridesB);
        if (broadcasted_strides_ptrs) UTILS_FREE(broadcasted_strides_ptrs);
        return NULL;
}

size_t* utils_broadcast_get_result_shape(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB, const OpType optype) {
    if (optype == DOT) return NULL;

    const int max_ndim = UTILS_GET_MAX(ndimA, ndimB);

    size_t* result_shape = (size_t*)malloc(max_ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(result_shape, stderr, alloc_failure);

    int offset = 0;
    if (optype == MATMUL) {
        result_shape[max_ndim-2] = shapeA[ndimA-2];
        result_shape[max_ndim-1] = shapeB[ndimB-1];
        offset = 2;
    }

    for (int i = max_ndim-offset-1; i >= 0; i--) {
        const int idxA = i - max_ndim + ndimA;
        const int idxB = i - max_ndim + ndimB;

        const int dimA = (idxA < 0) ? 1 : shapeA[idxA];
        const int dimB = (idxB < 0) ? 1 : shapeB[idxB];

        result_shape[i] = UTILS_GET_MAX(dimA, dimB);
    }

    if (optype == MATMUL) {
        if (ndimA == 1) {
            size_t** ptr = &result_shape;
            utils_erase_at_idx((void**)ptr, max_ndim-2, max_ndim, sizeof(size_t));
        } else if (ndimB == 1) {
            size_t** ptr = &result_shape;
            utils_erase_at_idx((void**)ptr, max_ndim-1, max_ndim, sizeof(size_t));
        } else if (ndimA == 1 && ndimB == 1) {
            fprintf(stderr, "utils_broadcast_get_result_shape: both ndims cannot be 1.");
            return NULL;
        }
    }
    
    return result_shape;

    alloc_failure:
        if (result_shape) UTILS_FREE(result_shape);
        return NULL;
}

size_t utils_broadcast_get_batch_size(const size_t* shape, const size_t ndim) {
    size_t batch_size = 1;
    if (ndim == 2) return batch_size;
    for (size_t i = 0; i < ndim-2; ++i) batch_size *= shape[i];
    return batch_size;
}