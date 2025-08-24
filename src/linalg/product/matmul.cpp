#include "matmul.h"

#include "tensor/tensor.h"
#include "utils/auxiliary.h"
#include "utils/broadcast.h"
#include "utils/metadata.h"

#include <stdlib.h>
#include <stdio.h>

Tensor* c_linalg_dot(const Tensor* a, const Tensor* b) {
    const size_t* shape = utils_broadcast_get_result_shape(
        a->shape, a->ndim, b->shape, b->ndim, true
    );

    const size_t ndim = UTILS_GET_MAX(a->ndim, b->ndim);

    const size_t size = utils_metadata_get_size(shape, ndim);

    float* _data = (float*)malloc(size * sizeof(float));

    if (!_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (size_t i = 0; i < size; i++) {
        *_data = a->_data[i] * b->_data[i];
    }

    return create_tensor(_data, shape, ndim);
} 