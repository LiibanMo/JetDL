#include "matmul.h"

#include "tensor/tensor.h"
#include "utils/auxiliary.h"
#include "utils/broadcast.h"

#include <stdlib.h>
#include <stdio.h>

Tensor* c_linalg_dot(const Tensor* a, const Tensor* b) {
    const size_t* shape = utils_broadcast_get_result_shape(
        a->shape, a->ndim, b->shape, b->ndim, DOT
    );

    const size_t ndim = 0;

    float* _data = (float*)malloc(sizeof(float));
    if (!_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    float* ptrA = a->_data; 
    float* ptrB = b->_data;

    for (size_t i = 0; i < a->size; i++) {
        *_data += *(ptrA + i) * *(ptrB + i);
    }

    ptrA = NULL;
    ptrB = NULL;

    return create_tensor(_data, shape, ndim);
} 