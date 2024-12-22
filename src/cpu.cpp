#include "tensor.h"

void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] + tensorB->data[idx];
    }
}

void scalar_add_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] + operand;
    }
}

void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] - tensorB->data[idx]; 
    }
}

void scalar_sub_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] - operand;
    }
}

void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] * tensorB->data[idx];
    }
}

void inner_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    result_data[0] = 1;
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[0] *= tensorA->data[idx] * tensorB->data[idx];
    }
}

void matmul_tensor_vector_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];

    for (int row_idx = 0; row_idx < M; row_idx++) {
        int sum = 0;
        for (int sum_idx = 0; sum_idx < N; sum_idx++) {
            sum += tensorA->data[N * row_idx + sum_idx] * tensorB->data[sum_idx];
        }
        result_data[row_idx] = sum;
    }
}

void matmul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (M, N) * (N, P) = (M, P)
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];
    const int P = tensorB->shape[1];

    for (int row_idx = 0; row_idx < M; row_idx++) {
        for (int col_idx = 0; col_idx < P; col_idx++) {
            double sum = 0;
            for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                sum += tensorA->data[N * row_idx + sum_idx] * tensorB->data[P * sum_idx + col_idx];
            }
            result_data[P * row_idx + col_idx] = sum;
        }
    }
}

void batch_matmul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (B, M, N) * (B, N, P) = (B, M, P)
    const int B = tensorA->shape[0];
    const int M = tensorA->shape[1];
    const int N = tensorA->shape[2];
    const int P = tensorB->shape[2];

    for (int batch_idx = 0; batch_idx < B; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                    sum += tensorA->data[M * N * batch_idx + N * row_idx + sum_idx] * tensorB->data[N * P * batch_idx + P * sum_idx + col_idx];
                }
                result_data[M * P * batch_idx + P * row_idx + col_idx] = sum;
            }
        }
    }
}

void scalar_mul_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] * operand;
    }
}

void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] / divisor;
    }
}