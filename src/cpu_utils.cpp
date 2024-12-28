#include <iostream>
#include "cpu_utils.h"

void vector_matmul_vector(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (N) @ (N) = (1)
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[0] += tensorA->data[idx] * tensorB->data[idx];
    }
}

void vector_matmul_matrix(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (N) @ (N, P) = (P)
    const int N = tensorA->shape[0];
    const int P = tensorB->shape[1];

    for (int col_idx = 0; col_idx < P; col_idx++) {
        double sum = 0;
        for (int sum_idx = 0; sum_idx < N; sum_idx++) {
            sum += tensorA->data[sum_idx] * tensorB->data[tensorB->strides[1]*sum_idx + col_idx];
        }
        result_data[col_idx] = sum;
    }
}

void vector_matmul_tensor(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (N) @ (B, N, P) = (B, P)
    const int B = tensorB->shape[0];
    const int N = tensorB->shape[1];
    const int P = tensorB->shape[2];

    for (int batch_idx = 0; batch_idx < B; batch_idx++) {
        for (int row_idx = 0; row_idx < P; row_idx++) {
            double sum = 0;
            for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                sum += tensorA->data[sum_idx] * tensorB->data[tensorB->strides[2]*batch_idx + tensorB->strides[1]*sum_idx + row_idx];
            }
            result_data[P*batch_idx + row_idx] = sum;
        }
    }
}

void matrix_matmul_vector(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (M, N) @ (N) = (M)
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];

    for (int row_idx = 0; row_idx < M; row_idx++) {
        double sum = 0;
        for (int sum_idx = 0; sum_idx < N; sum_idx++) {
            sum += tensorA->data[tensorA->strides[0]*row_idx + sum_idx] * tensorB->data[sum_idx];
        }
        result_data[row_idx] = sum;
    }
}

void matrix_matmul_matrix(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (M, N) @ (N, P) = (M, P)
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];
    const int P = tensorB->shape[1];

    for (int row_idx = 0; row_idx < M; row_idx++) {
        for (int col_idx = 0; col_idx < P; col_idx++) {
            double sum = 0;
            for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                sum += tensorA->data[tensorA->strides[0]*row_idx + sum_idx] * tensorB->data[tensorB->strides[0] * sum_idx + col_idx];
            }
            result_data[P*row_idx + col_idx] = sum;
        }
    }
}

void matrix_matmul_tensor(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (M, N) @ (B, N, P) = (B, M, P)
    const int B = tensorB->shape[0];
    const int M = tensorA->shape[0];
    const int P = tensorB->shape[2];
    const int N = tensorA->shape[1];

    for (int batch_idx = 0; batch_idx < B; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                    sum += tensorA->data[tensorA->strides[1]*row_idx + sum_idx] * tensorB->data[tensorB->strides[2]*batch_idx + tensorB->strides[1]*sum_idx + col_idx];
                }
                result_data[M*P*batch_idx + P*row_idx + col_idx] = sum;
            }
        }
    }
}

void tensor_matmul_vector(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (B, M, N) @ (N) = (B, M)
    const int B = tensorA->shape[0];
    const int M = tensorA->shape[1];
    const int N = tensorA->shape[2];

    for (int batch_idx = 0; batch_idx < B; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            double sum = 0;
            for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                sum += tensorA->data[tensorA->strides[2]*batch_idx + tensorA->strides[1]*row_idx + sum_idx] * tensorB->data[sum_idx];
            }
            result_data[M*batch_idx + row_idx] = sum;
        }
    }
}

void tensor_matmul_matrix(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (B, M, N) @ (N, P) = (B, M, P)
    const int B = tensorA->shape[0];
    const int M = tensorA->shape[1];
    const int N = tensorA->shape[2];
    const int P = tensorB->shape[1];

    for (int batch_idx = 0; batch_idx < B; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                    sum += tensorA->data[tensorA->strides[2]*batch_idx + tensorA->strides[1]*row_idx + sum_idx] * tensorB->data[tensorB->strides[1]*sum_idx + col_idx];
                }
                result_data[M*P*batch_idx + P*row_idx + col_idx] = sum;
            }
        }
    }
}

void tensor_matmul_tensor(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (B, M, N) @ (B, N, P) = (B, M, P)
    const int B = tensorA->shape[0];
    const int M = tensorA->shape[1];
    const int P = tensorB->shape[2];
    const int N = tensorA->shape[2];

    for (int batch_idx = 0; batch_idx < B; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                    sum += tensorA->data[tensorA->strides[2]*batch_idx + tensorA->strides[1]*row_idx + sum_idx] * tensorB->data[tensorB->strides[2]*batch_idx + tensorB->strides[1]*sum_idx + col_idx];
                }
                result_data[M*P*batch_idx + P*row_idx + col_idx] = sum;
            } 
        }
    }
}