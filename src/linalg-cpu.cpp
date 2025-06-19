#include <omp.h>
#include <cblas.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "lib.h"

void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data) {
    const int N = tensorA->shape[0];
    result_data[0] = cblas_sdot(N, tensorA->data, 1, tensorB->data, 1);
}

void matmul_2d_2d_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data) {
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];
    const int P = tensorB->shape[1];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, P, N, 
                1.0f, tensorA->data, N, 
                tensorB->data, P, 
                0.0f, result_data, P);
}

void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data){
    const int M = tensorA->shape[tensorA->ndim-2];
    const int N = tensorA->shape[tensorA->ndim-1];
    const int P = tensorB->shape[tensorB->ndim-1];

    const int max_ndim = std::max(tensorA->ndim, tensorB->ndim);

    int* stridesA = (tensorA->ndim < max_ndim) ? (int*)malloc(max_ndim * sizeof(int)) : tensorA->strides;
    int* stridesB = (tensorB->ndim < max_ndim) ? (int*)malloc(max_ndim * sizeof(int)) : tensorB->strides;
    
    if ((tensorA->ndim < max_ndim && !stridesA) || (tensorB->ndim < max_ndim && !stridesB)) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    
    int total_num_matrices = 1;
    int* shape_ptr = (tensorA->ndim >= tensorB->ndim) ? tensorA->shape : tensorB->shape;
    for (int idx = 0; idx < max_ndim-2; idx++) {
        total_num_matrices *= shape_ptr[idx];
    }

    if (tensorA->ndim < max_ndim) {
        for (int idx = 0; idx < tensorB->ndim; idx++) {
            int idxA = idx - tensorB->ndim + tensorA->ndim;
            stridesA[idx] = (idxA < 0) ? 0 : tensorA->strides[idxA];
        }
    } else if (tensorB->ndim < max_ndim) {
        for (int idx = 0; idx < tensorA->ndim; idx++) {
            int idxB = idx - tensorA->ndim + tensorB->ndim;
            stridesB[idx] = (idxB < 0) ? 0 : tensorB->strides[idxB];
        }
    }

    const int NUM_ITERS = total_num_matrices * M * P * N;
    const int NUM_THREADS = std::min(omp_get_max_threads(), 
                                   std::max(1, NUM_ITERS / (256 * 1024)));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int batch_idx = 0; batch_idx < total_num_matrices; batch_idx++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    M, P, N, 
                    1.0f, 
                    &tensorA->data[stridesA[max_ndim-3]*batch_idx], N, 
                    &tensorB->data[stridesB[max_ndim-3]*batch_idx], P, 
                    0.0f, 
                    &result_data[M*P*batch_idx], P);
    }

    if (tensorA->ndim < max_ndim && stridesA) {
        free(stridesA);
    }
    if (tensorB->ndim < max_ndim && stridesB) {
        free(stridesB);
    }
}

void outer_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data) {
    const int N = tensorA->size;
    const int M = tensorB->size;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result_data[i * M + j] = tensorA->data[i] * tensorB->data[j];
        }
    }
}