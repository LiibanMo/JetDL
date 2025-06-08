#include <cstddef>
#include <iostream>
#include <cmath>
#include <omp.h>
#include "lib.h"


void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int N = tensorA->shape[0];
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(N / 10, 1));

    double sum = 0;
    #pragma omp parallel for reduction(+:sum) num_threads(NUM_THREADS)
    for (int sum_idx = 0; sum_idx < N; sum_idx++) {
        sum += tensorA->data[sum_idx] * tensorB->data[sum_idx];
    }
    result_data[0] = sum;
}

void matmul_2d_2d_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];
    const int P = tensorB->shape[1];

    const int NUM_ITERS = M * P * N;
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(NUM_ITERS / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int row_idx = 0; row_idx < M; row_idx++) {
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int col_idx = 0; col_idx < P; col_idx++) {
            double sum = 0;
            for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                int idxA = tensorA->strides[tensorA->ndim-2]*row_idx + tensorA->strides[tensorA->ndim-1]*sum_idx;
                int idxB = tensorB->strides[tensorB->ndim-2]*sum_idx + tensorB->strides[tensorB->ndim-1]*col_idx;
                sum += tensorA->data[idxA] * tensorB->data[idxB]; 
            }
            int idx_result_data = P*row_idx + col_idx;
            result_data[idx_result_data] = sum;
        }
    }
}

void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data){
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
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(NUM_ITERS / 10, 1));

    #pragma omp parallel for collapse(3) num_threads(NUM_THREADS)
    for (int batch_idx = 0; batch_idx < total_num_matrices; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                #pragma omp parallel for reduction(+:sum)
                for (int sum_idx = 0; sum_idx < N; sum_idx++) { 
                    int idxA = stridesA[max_ndim-3]*batch_idx + stridesA[max_ndim-2]*row_idx + stridesA[max_ndim-1]*sum_idx;
                    int idxB = stridesB[max_ndim-3]*batch_idx + stridesB[max_ndim-2]*sum_idx + stridesB[max_ndim-1]*col_idx;
                    sum += tensorA->data[idxA] * tensorB->data[idxB];
                }
                int idx_result = M*P*batch_idx + P*row_idx + col_idx;
                result_data[idx_result] = sum;
            }
        }
    }

    if (tensorA->ndim < max_ndim) {
        free(stridesA);
    } else { // tensorB->ndim < max_ndim
        free(stridesB);
    }
}

void outer_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size * tensorB->size / 10, 1));

    #pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
    for (int i = 0; i < tensorA->size; i++) {
        for (int j = 0; j < tensorB->size; j++) {
            result_data[tensorB->size * i + j] = tensorA->data[i] * tensorB->data[j];
        }
    }
}