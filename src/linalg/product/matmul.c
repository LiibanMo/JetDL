#include "matmul.h"
#include "kernel.h"

#include "tensor/tensor.h"
#include "utils/auxiliary.h"
#include "utils/broadcast.h"
#include "utils/metadata.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor* c_linalg_dot(const Tensor* a, const Tensor* b) {
    float* result_data = (float*)calloc(1, sizeof(float));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (size_t i = 0; i < a->size; i++) {
        result_data[0] += a->_data[i] * b->_data[i];
    }

    return create_tensor(result_data, NULL, 0);
} 

Tensor* c_linalg_matvec(const Tensor* a, const Tensor* b) {
    // (..., M, N) @ (N)
    size_t* shape = utils_broadcast_get_result_shape(
        a->shape, a->ndim, b->shape, b->ndim, MATMUL
    );
    
    const size_t ndim = a->ndim - 1;

    const size_t M = a->shape[a->ndim - 2];
    const size_t N = b->shape[0];

    const size_t DATA1_ROWS = UTILS_NEXT_MULTIPLE(M, BLOCK_N_ROWS);
    const size_t BATCH_SIZE = utils_broadcast_get_batch_size(a->shape, a->ndim);

    const size_t DATA1_MAT_SIZE = DATA1_ROWS * N;
    const size_t DATA2_MAT_SIZE = N * BLOCK_N_COLS;
    const size_t RESULT_MAT_SIZE = DATA1_ROWS * BLOCK_N_COLS;

    float* result_matrix = (float*)calloc(RESULT_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_matrix, stderr, alloc_failure);

    float* data1_matrix = (float*)calloc(DATA1_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data1_matrix, stderr, alloc_failure);

    float* data2_matrix = (float*)calloc(DATA2_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data2_matrix, stderr, alloc_failure);

    const size_t RESULT_DATA_SIZE = a->size / b->size;
    float* result_data = (float*)malloc(RESULT_DATA_SIZE * sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_data, stderr, alloc_failure);

    for (size_t i = 0; i < N; i++) {
        data2_matrix[i * BLOCK_N_COLS] = b->_data[i];
    }
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        const size_t a_batch_stride = (a->ndim > 2) ? a->strides[a->ndim - 3] : 0;
        const size_t idxA = batch * a_batch_stride;
        memcpy(&data1_matrix[0], &a->_data[idxA], M * N * sizeof(float));
        for (size_t x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
            c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, 0, BLOCK_N_COLS, N);
        }
        for (size_t i = 0; i < M; i++) {
            result_data[batch * M + i] = result_matrix[i * BLOCK_N_COLS];
        }
    }

    UTILS_FREE(result_matrix);
    UTILS_FREE(data1_matrix);
    UTILS_FREE(data2_matrix);

    return create_tensor(result_data, shape, ndim);

    alloc_failure:
        if (result_matrix) UTILS_FREE(result_matrix);
        if (data1_matrix) UTILS_FREE(data1_matrix);
        if (data2_matrix) UTILS_FREE(data2_matrix);
        return NULL;
}

Tensor* c_linalg_vecmat(const Tensor* a, const Tensor* b) {
    // (N) @ (..., N, P)
    size_t* shape = utils_broadcast_get_result_shape(
        a->shape, a->ndim, b->shape, b->ndim, MATMUL
    );

    const size_t ndim = b->ndim - 1;

    const size_t N = a->shape[0];
    const size_t P = b->shape[b->ndim - 1];

    const size_t DATA2_COLS = UTILS_NEXT_MULTIPLE(P, BLOCK_N_COLS);
    const size_t BATCH_SIZE = utils_broadcast_get_batch_size(b->shape, b->ndim);

    const size_t DATA1_MAT_SIZE = BLOCK_N_ROWS * N;
    const size_t DATA2_MAT_SIZE = N * DATA2_COLS;
    const size_t RESULT_MAT_SIZE = BLOCK_N_ROWS * DATA2_COLS;

    float* result_matrix = (float*)calloc(RESULT_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_matrix, stderr, alloc_failure);

    float* data1_matrix = (float*)calloc(DATA1_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data1_matrix, stderr, alloc_failure);

    float* data2_matrix = (float*)calloc(DATA2_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data2_matrix, stderr, alloc_failure);

    const size_t RESULT_DATA_SIZE = b->size / a->size;
    float* result_data = (float*)malloc(RESULT_DATA_SIZE * sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_data, stderr, alloc_failure);

    memcpy(&data1_matrix[0], &a->_data[0], N * sizeof(float)); 
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t i = 0; i < N; i++) {
            const size_t b_batch_stride = (b->ndim > 2) ? b->strides[b->ndim - 3] : 0;
            const size_t idxB = batch * b_batch_stride + i * b->strides[b->ndim-2];
            memcpy(&data2_matrix[i * DATA2_COLS], &b->_data[idxB], P * sizeof(float));
        }
        for (size_t y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
            c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, 0, y, DATA2_COLS, N);
        }
        memcpy(&result_data[batch * P], &result_matrix[0], P * sizeof(float));
    }

    UTILS_FREE(data1_matrix);
    UTILS_FREE(data2_matrix);
    UTILS_FREE(result_matrix);

    return create_tensor(result_data, shape, ndim);

    alloc_failure:
        if (result_matrix) free(result_matrix);
        if (data1_matrix) free(data1_matrix);
        if (data2_matrix) free(data2_matrix);
        if (result_data) free(result_data);
        return NULL;
}

Tensor* c_linalg_matmul(const Tensor* a, const Tensor* b) {
    // (..., M, N) @ (..., N, P)
    size_t* shape = utils_broadcast_get_result_shape(
        a->shape, a->ndim, b->shape, b->ndim, MATMUL
    );

    const size_t ndim = UTILS_GET_MAX(a->ndim, b->ndim);
    const size_t M = a->shape[a->ndim - 2];
    const size_t N = a->shape[a->ndim - 1];
    const size_t P = b->shape[b->ndim - 1];
    
    const size_t BATCH_SIZE = utils_broadcast_get_batch_size(shape, ndim);
    
    const size_t DATA1_ROWS = UTILS_NEXT_MULTIPLE(M, BLOCK_N_ROWS);
    const size_t DATA2_COLS = UTILS_NEXT_MULTIPLE(P, BLOCK_N_COLS);

    const size_t RESULT_MAT_SIZE = DATA1_ROWS * DATA2_COLS;
    const size_t DATA1_MAT_SIZE = DATA1_ROWS * N;
    const size_t DATA2_MAT_SIZE = N * DATA2_COLS;

    size_t** strides_ptrs = utils_broadcast_get_strides(
       a->shape, a->ndim, b->shape, b->ndim, MATMUL
    );

    size_t* stridesA = strides_ptrs[0];
    size_t* stridesB = strides_ptrs[1];

    size_t* idxsA = utils_populate_linear_idxs(shape, stridesA, ndim, MATMUL);
    size_t* idxsB = utils_populate_linear_idxs(shape, stridesB, ndim, MATMUL);

    float* result_matrix = (float*)calloc(RESULT_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_matrix, stderr, alloc_failure);

    float* data1_matrix = (float*)calloc(DATA1_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data1_matrix, stderr, alloc_failure);

    float* data2_matrix = (float*)calloc(DATA2_MAT_SIZE, sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(data2_matrix, stderr, alloc_failure);

    const size_t RESULT_DATA_SIZE = utils_metadata_get_size(shape, ndim); 
    float* result_data = (float*)malloc(RESULT_DATA_SIZE * sizeof(float));
    UTILS_CHECK_ALLOC_FAILURE(result_data, stderr, alloc_failure);

    size_t count = 1;
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        const size_t idxA = idxsA[batch];
        memcpy(&data1_matrix[0], &a->_data[idxA], M * N * sizeof(float));
        for (size_t i = 0; i < N; i++) {
            const size_t idxB = idxsB[batch] + i * b->strides[b->ndim-2];
            memcpy(&data2_matrix[i * DATA2_COLS], &b->_data[idxB], P * sizeof(float));
        }
        for (size_t x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
            for (size_t y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
                c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, y, DATA2_COLS, N);
            }
        }
        for (size_t i = 0; i < M; i++) {
            const size_t idx = batch * M * P + i * P;
            memcpy(&result_data[idx], &result_matrix[i * DATA2_COLS], P * sizeof(float));
        }
    }

    UTILS_FREE(stridesA);
    UTILS_FREE(stridesB);
    UTILS_FREE(strides_ptrs);

    UTILS_FREE(idxsA);
    UTILS_FREE(idxsB);

    UTILS_FREE(result_matrix);
    UTILS_FREE(data1_matrix);
    UTILS_FREE(data2_matrix);

    return create_tensor(result_data, shape, ndim);

    alloc_failure:
        if (result_matrix) UTILS_FREE(result_matrix);
        if (data1_matrix) UTILS_FREE(data1_matrix);
        if (data2_matrix) UTILS_FREE(data2_matrix);
        if (result_data) UTILS_FREE(result_data);
        return NULL;
}