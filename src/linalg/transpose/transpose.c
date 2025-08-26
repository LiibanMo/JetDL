#include "transpose.h"
#include "tensor/tensor.h"
#include "utils/auxiliary.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor* c_linalg_T(const Tensor* a) {
    Tensor* result_tensor = create_tensor(a->_data, a->shape, a->ndim);
    
    const size_t START = 0;
    const size_t END = result_tensor->ndim;

    utils_reverse((void**)&result_tensor->shape, START, END, sizeof(size_t));
    utils_reverse((void**)&result_tensor->strides, START, END, sizeof(size_t));

    if (memcmp(&a->strides[0], &result_tensor->strides[0], a->ndim * sizeof(size_t)) != 0) {
        result_tensor->is_contiguous = false;
    }

    return result_tensor;
}

Tensor* c_linalg_mT(const Tensor* a) {
    Tensor* result_tensor = create_tensor(a->_data, a->shape, a->ndim);
    const size_t START = result_tensor->ndim - 2;
    const size_t END = result_tensor->ndim;
    
    utils_reverse((void**)&result_tensor->shape, START, END, sizeof(size_t));
    utils_reverse((void**)&result_tensor->strides, START, END, sizeof(size_t));
    
    if (memcmp(&a->strides[0], &result_tensor->strides[0], a->ndim * sizeof(size_t)) != 0) {
        result_tensor->is_contiguous = false;
    }

    return result_tensor;
}