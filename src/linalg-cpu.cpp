#include <iostream>
#include <cmath>
#include <omp.h>
#include "string.h"
#include "tensor.h"


void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int N = tensorA->shape[0];
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(N / 10, 1));

    double sum = 0;
    #pragma omp parallel for reduction(+:sum), num_threads(NUM_THREADS)
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

    #pragma omp parallel for, collapse(2), num_threads(NUM_THREADS)
    for (int row_idx = 0; row_idx < M; row_idx++) {
        for (int col_idx = 0; col_idx < P; col_idx++) {
            double sum = 0;
            #pragma omp parallel for reduction(+:sum)
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

void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, char broadcasted[]){
    const int M = tensorA->shape[tensorA->ndim-2];
    const int N = tensorA->shape[tensorA->ndim-1];
    const int P = tensorB->shape[tensorB->ndim-1];

    int total_num_matrices = 1;
    if (strcmp(broadcasted, "A") == 0) {
        for (int idx = 0; idx < tensorB->ndim-2; idx++) {
            total_num_matrices *= tensorB->shape[idx];
        }
    } else if (strcmp(broadcasted, "B") == 0) {
        for (int idx = 0; idx < tensorA->ndim-2; idx++) {
            total_num_matrices *= tensorA->shape[idx];
        }
    } else {
        fprintf(stderr, "Incorrect character inputted. Should be A or B");
        exit(1);
    }

    const int NUM_ITERS = total_num_matrices * M * P * N;
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(NUM_ITERS / 10, 1));

    #pragma omp parallel for, collapse(3), num_threads(NUM_THREADS)
    for (int batch_idx = 0; batch_idx < total_num_matrices; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                #pragma omp parallel for reduction(+:sum)
                for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                    int idxA = tensorA->strides[tensorA->ndim-3]*batch_idx + tensorA->strides[tensorA->ndim-2]*row_idx + tensorA->strides[tensorA->ndim-1]*sum_idx;
                    int idxB = tensorB->strides[tensorB->ndim-3]*batch_idx + tensorB->strides[tensorB->ndim-2]*sum_idx + tensorB->strides[tensorB->ndim-1]*col_idx;
                    sum += tensorA->data[idxA] * tensorB->data[idxB];
                }
                int idx_result = M*P*batch_idx + P*row_idx + col_idx;
                result_data[idx_result] = sum;
            }
        }
    }
}

void outer_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size * tensorB->size / 10, 1));

    #pragma omp parallel for, collapse(2), num_threads(NUM_THREADS)
    for (int i = 0; i < tensorA->size; i++) {
        for (int j = 0; j < tensorB->size; j++) {
            result_data[tensorB->size * i + j] = tensorA->data[i] * tensorB->data[j];
        }
    }
}