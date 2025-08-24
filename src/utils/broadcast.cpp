#include "broadcast.h"
#include "auxiliary.h"
#include "metadata.h"

#include <cstddef>
#include <stdlib.h>
#include <stdio.h>

size_t** utils_broadcast_get_strides(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB, const bool matmul) {
    const size_t max_ndim = UTILS_GET_MAX(ndimA, ndimB);

    size_t* broadcasted_stridesA = (size_t*)calloc(max_ndim, sizeof(size_t));
    size_t* broadcasted_stridesB = (size_t*)calloc(max_ndim, sizeof(size_t));
    if (!broadcasted_stridesA || !broadcasted_stridesB) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    size_t** broadcasted_strides_ptrs = (size_t**)malloc(2 * sizeof(size_t*));
    if (!broadcasted_strides_ptrs) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    broadcasted_strides_ptrs[0] = broadcasted_stridesA;
    broadcasted_strides_ptrs[1] = broadcasted_stridesB;

    size_t* stridesA = utils_metadata_get_strides(shapeA, ndimA);
    size_t* stridesB = utils_metadata_get_strides(shapeB, ndimB);

    const size_t offset = matmul ? 2 : 0;

    for (size_t i = max_ndim - offset - 1; i >= 0; i--) {
        const size_t idxA = i - max_ndim + ndimA;
        const size_t idxB = i - max_ndim + ndimB;

        const size_t dimA = (idxA < 0) ? 1 : shapeA[idxA];
        const size_t dimB = (idxB < 0) ? 1 : shapeB[idxB];

        broadcasted_stridesA[i] = (dimA == 1 && dimA < dimB) ? 0 : stridesA[idxA];
        broadcasted_stridesB[i] = (dimB == 1 && dimB < dimA) ? 0 : stridesB[idxB];
    }

    free(stridesA);
    stridesA = NULL;
    free(stridesB);
    stridesB = NULL;

    return broadcasted_strides_ptrs;
}

size_t* utils_broadcast_get_result_shape(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB, const bool matmul) {
    const int max_ndim = UTILS_GET_MAX(ndimA, ndimB);

    size_t* result_shape = (size_t*)malloc(max_ndim * sizeof(size_t));
    if (!result_shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    int offset = 0;
    if (matmul) {
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

    // Assumes only one operand can be a vector
    if (matmul) {
        if (ndimA == 1) {
            
        } else if (ndimB == 1) {
            size_t** ptr = &result_shape;
            utils_erase_at_idx(reinterpret_cast<void**>(&result_shape), max_ndim-1, max_ndim, sizeof(size_t));
        }
    }
    
    return result_shape;
}

size_t utils_broadcast_get_batch_size(const size_t* shape, const size_t ndim) {
    if (ndim < 3) {
        fprintf(stderr, "ndim is less than 3; no batch dimensions.");
        return 0;
    }
    size_t batch_size = 1;
    for (size_t i = 2; i < ndim; ++i) {
        batch_size *= shape[i];
    }
    return batch_size;
}