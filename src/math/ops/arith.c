#include "arith.h"
#include "math/ops/kernel.h"
#include "tensor/tensor.h"
#include "utils/auxiliary.h"
#include "utils/broadcast.h"
#include "utils/metadata.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor* c_math_ops(const Tensor* a, const Tensor* b, ArithType arith_type) {
    const size_t ndim = UTILS_GET_MAX(a->ndim, b->ndim);

    size_t* shape = utils_broadcast_get_result_shape(
        a->shape, a->ndim, b->shape, b->ndim, ARITHMETIC
    );

    size_t** strides_ptrs = utils_broadcast_get_strides(
        a->shape, a->ndim, b->shape, b->ndim, ARITHMETIC
    );

    size_t* stridesA = strides_ptrs[0];
    size_t* stridesB = strides_ptrs[1];

    size_t* idxsA = utils_populate_linear_idxs(shape, stridesA, ndim, ARITHMETIC);
    UTILS_FREE(stridesA);
    
    size_t* idxsB = utils_populate_linear_idxs(shape, stridesB, ndim, ARITHMETIC);
    UTILS_FREE(stridesB);
    
    UTILS_FREE(strides_ptrs);
    
    const size_t NA = a->shape[a->ndim - 1];
    const size_t NB = b->shape[b->ndim - 1];
    const size_t N = UTILS_GET_MAX(NA, NB);
    
    const size_t DATA_VEC_SIZE = UTILS_NEXT_MULTIPLE(N, BLOCK_N_COLS);
    
    float* result_vec = (float*)calloc(DATA_VEC_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_vec, stderr, alloc_failure);

    float* data1_vec = (float*)calloc(DATA_VEC_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data1_vec, stderr, alloc_failure);

    float* data2_vec = (float*)calloc(DATA_VEC_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data2_vec, stderr, alloc_failure);

    const size_t RESULT_DATA_SIZE = utils_metadata_get_size(shape, ndim);
    float* result_data = (float*)malloc(RESULT_DATA_SIZE * sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_data, stderr, alloc_failure);

    void (*kernel)(const float* a, const float* b, float* c, const size_t N); 
    if (arith_type == ADD) kernel = c_add_cpu;
    if (arith_type == SUB) kernel = c_sub_cpu;
    if (arith_type == MUL) kernel = c_mul_cpu;
    if (arith_type == DIV) kernel = c_div_cpu;

    const size_t TOTAL_NUM_ROWS = RESULT_DATA_SIZE / shape[ndim - 1];
    
    if (NA == NB) {
        for (size_t row = 0; row < TOTAL_NUM_ROWS; row++) { 
            memcpy(data1_vec, a->_data + idxsA[row], N * sizeof(float));
            memcpy(data2_vec, b->_data + idxsB[row], N * sizeof(float));
            (*kernel)(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
            memcpy(&result_data[row * N], &result_vec[0], N * sizeof(float));
        }
    } else if (NA < NB && NA == 1) {
        for (size_t row = 0; row < TOTAL_NUM_ROWS; row++) {
            utils_fill(data1_vec, &a->_data[0], N, sizeof(float));
            memcpy(data2_vec, b->_data + idxsB[row], N * sizeof(float));
            (*kernel)(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
            memcpy(result_data + row * N, result_vec, N * sizeof(float));
        }
    } else if (NA > NB && NA == 1) {
        for (size_t row = 0; row < TOTAL_NUM_ROWS; row++) {
            memcpy(data1_vec, a->_data + idxsA[row], N * sizeof(float));
            utils_fill(data2_vec, &b->_data[0], N, sizeof(float));
            (*kernel)(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
            memcpy(result_data + row * N, result_vec, N * sizeof(float));
        }
    }

    UTILS_FREE(idxsA);
    UTILS_FREE(idxsB);

    UTILS_FREE(result_vec);
    UTILS_FREE(data1_vec);
    UTILS_FREE(data2_vec);

    return create_tensor(result_data, shape, ndim);

    alloc_failure:
        if (shape) UTILS_FREE(shape);
        if (stridesA) UTILS_FREE(stridesA);
        if (stridesB) UTILS_FREE(stridesB);
        if (strides_ptrs) UTILS_FREE(strides_ptrs);
        if (result_vec) UTILS_FREE(result_vec);
        if (result_data) UTILS_FREE(result_data);
        return NULL;
}