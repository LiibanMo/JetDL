#include "transpose.h"
#include "tensor/tensor.h"
#include "utils/auxiliary.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor* c_linalg_T(const Tensor* a) {
    const size_t ndim = a->ndim;

    size_t* shape = (size_t*)malloc(ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(shape, stderr, alloc_failure);

    size_t* strides = (size_t*)malloc(ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(strides, stderr, alloc_failure);

    for (size_t i = 0; i < ndim; i++) {
        shape[i] = a->shape[ndim - 1 - i];
        strides[i] = a->strides[ndim -1 - i];
    }
    
    float* _data = (float*)malloc(a->size * sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(_data, stderr, alloc_failure);
    memcpy(_data, a->_data, a->size * sizeof(float));
    
    const size_t START = 0;
    const size_t END = ndim;
    Tensor* result_tensor = create_tensor(_data, shape, ndim);

    size_t* strides_temp_ptr = result_tensor->strides;
    result_tensor->strides = strides;
    if (strides_temp_ptr) UTILS_FREE(strides_temp_ptr);

    if (memcmp(a->strides, result_tensor->strides, ndim * sizeof(size_t)) != 0) {
        result_tensor->is_contiguous = false;
    }

    return result_tensor;

    alloc_failure:
            if (shape) UTILS_FREE(shape);
            if (strides) UTILS_FREE(strides);
            if (_data) UTILS_FREE(_data);
        return NULL;
}

Tensor* c_linalg_mT(const Tensor* a) {
    const size_t ndim = a->ndim;
    size_t* shape = (size_t*)malloc(ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(shape, stderr, alloc_failure);
   
    size_t* strides = (size_t*)malloc(ndim * sizeof(size_t));
    UTILS_CHECK_ALLOC_FAILURE(strides, stderr, alloc_failure);

    for (size_t i = 0; i < ndim - 2; i++) {
        shape[i] = a->shape[i];
        strides[i] = a->strides[i];
    }

    shape[ndim - 2] = a->shape[ndim - 1];
    shape[ndim - 1] = a->shape[ndim - 2];

    strides[ndim - 2] = a->strides[ndim - 1];
    strides[ndim - 1] = a->strides[ndim - 2];

    const size_t size = a->size;
    float* _data = (float*)malloc(size * sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(_data, stderr, alloc_failure);
    memcpy(_data, a->_data, size * sizeof(float));

    Tensor* result_tensor = create_tensor(_data, shape, ndim);

    size_t* strides_temp_ptr = result_tensor->strides;
    result_tensor->strides = strides;
    if (strides_temp_ptr) UTILS_FREE(strides_temp_ptr);
    
    if (memcmp(a->strides, result_tensor->strides, a->ndim * sizeof(size_t)) != 0) {
        result_tensor->is_contiguous = false;
    }

    return result_tensor;

    alloc_failure:
        if (shape) UTILS_FREE(shape);
        if (strides) UTILS_FREE(strides);
        if (_data) UTILS_FREE(_data);
        return NULL;
}