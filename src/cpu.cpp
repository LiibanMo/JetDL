#include <iostream>
#include "string.h"
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

void scalar_mul_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] * operand;
    }
}

void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] * tensorB->data[idx];
    }
}

void matmul_2d_2d_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];
    const int P = tensorB->shape[1];

    for (int row_idx = 0; row_idx < M; row_idx++) {
        for (int col_idx = 0; col_idx < P; col_idx++) {
            double sum = 0;
            for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                int strideA = tensorA->strides[0]*row_idx + sum_idx;
                int strideB = tensorB->strides[0]*sum_idx + col_idx;
                sum += tensorA->data[strideA] * tensorB->data[strideB]; 
            }
            int stride_result_data = P*row_idx + col_idx;
            result_data[stride_result_data] = sum;
        }
    }
}

void matmul_prepended_1d_a_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int N = tensorB->shape[0];
    const int P = tensorB->shape[1];

    for (int col_idx = 0; col_idx < P; col_idx++) {
        double sum = 0;
        for (int sum_idx; sum_idx < N; sum_idx++) {
            int strideA = sum_idx;
            int strideB = tensorB->strides[0]*sum_idx + col_idx;
            sum += tensorA->data[strideA] * tensorB->data[strideB];
        }
        int stride_result_data = col_idx;
        result_data[stride_result_data] = sum;
    }
}

void matmul_appended_1d_b_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    // (M, N) @ (N) = (M)
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];

    for (int row_idx = 0; row_idx < M; row_idx++) {
        double sum = 0;
        for (int sum_idx = 0; sum_idx < N; sum_idx++) {
            int strideA = tensorA->strides[0] * row_idx + sum_idx;
            int strideB = sum_idx;
            sum += tensorA->data[strideA] * tensorB->data[strideB];
        }
        int stride_result_data = row_idx;
        result_data[stride_result_data] = sum;
    }
}

void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, char broadcasted_matrix[]){
    const int M = tensorA->shape[tensorA->ndim-2];
    const int N = tensorA->shape[tensorA->ndim-1];
    const int P = tensorB->shape[tensorB->ndim-1];

    int total_num_matrices = 1;
    if (strcmp(broadcasted_matrix, "A") == 0) {
        for (int idx = 0; idx < tensorB->ndim-2; idx++) {
            total_num_matrices *= tensorB->shape[idx];
        }
    } else if (strcmp(broadcasted_matrix, "B") == 0) {
        for (int idx = 0; idx < tensorA->ndim-2; idx++) {
            total_num_matrices *= tensorA->shape[idx];
        }
    } else {
        fprintf(stderr, "Incorrect character inputted. Should be A or B");
        exit(1);
    }

    for (int batch_idx = 0; batch_idx < total_num_matrices; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                    int strideA = tensorA->strides[tensorA->ndim-3]*batch_idx + tensorA->strides[tensorA->ndim-2]*row_idx + sum_idx;
                    int strideB = tensorB->strides[tensorB->ndim-3]*batch_idx + tensorB->strides[tensorB->ndim-2]*sum_idx + col_idx;
                    sum += tensorA->data[strideA] * tensorB->data[strideB];
                }
                result_data[M*P*batch_idx + P*row_idx + col_idx] = sum;
            }
        }
    }
}

void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int N = tensorA->shape[0];

    for (int sum_idx = 0; sum_idx < N; sum_idx++) {
        result_data[0] += tensorA->data[sum_idx] * tensorB->data[sum_idx];
    }
}

void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] / divisor;
    }
}