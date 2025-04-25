#include <iostream>
#include <cmath>
#include "string.h"
#include "tensor.h"


void assign_tensor_data_cpu(Tensor* tensor, double* result_data) {
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[idx] = tensor->data[idx];
    }
}

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

void add_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* stridesA = (int*)malloc(ndim * sizeof(int));
    if (!stridesA) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int* stridesB = (int*)malloc(ndim * sizeof(int));
    if (!stridesB) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int strideA = 1, strideB = 1;
    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        stridesA[idx] = (dimA == broadcasted_shape[idx]) ? strideA : 0;
        stridesB[idx] = (dimB == broadcasted_shape[idx]) ? strideB : 0;

        strideA *= (dimA == broadcasted_shape[idx]) ? dimA : 1;
        strideB *= (dimB == broadcasted_shape[idx]) ? dimB : 1;
    }

    for (int i = 0; i < broadcasted_size; i++) {
        int idx_result = i;
        int idxA = 0;
        int idxB = 0;
        for (int j = ndim-1; j >= 0; j--) {
            int pos = idx_result % broadcasted_shape[j];
            idx_result /= broadcasted_shape[j];
            idxA += stridesA[j] * pos;
            idxB += stridesB[j] * pos; 
        }
        result_data[i] = tensorA->data[idxA] + tensorB->data[idxB];
    }

    free(stridesA);
    free(stridesB);
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

void sub_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* stridesA = (int*)malloc(ndim * sizeof(int));
    if (!stridesA) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int* stridesB = (int*)malloc(ndim * sizeof(int));
    if (!stridesB) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int strideA = 1, strideB = 1;
    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        stridesA[idx] = (dimA == broadcasted_shape[idx]) ? strideA : 0;
        stridesB[idx] = (dimB == broadcasted_shape[idx]) ? strideB : 0;

        strideA *= (dimA == broadcasted_shape[idx]) ? dimA : 1;
        strideB *= (dimB == broadcasted_shape[idx]) ? dimB : 1;
    } 

    for (int i = 0; i < broadcasted_size; i++) {
        int idx_result = i;
        int idxA = 0;
        int idxB = 0;
        for (int j = ndim-1; j >= 0; j--) {
            int pos = idx_result % broadcasted_shape[j];
            idx_result /= broadcasted_shape[j];
            idxA += stridesA[j] * pos;
            idxB += stridesB[j] * pos;  
        }
        result_data[i] = tensorA->data[idxA] - tensorB->data[idxB];
    }

    free(stridesA);
    free(stridesB);
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

void mul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* stridesA = (int*)malloc(ndim * sizeof(int));
    if (!stridesA) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int* stridesB = (int*)malloc(ndim * sizeof(int));
    if (!stridesB) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int strideA = 1, strideB = 1;
    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        stridesA[idx] = (dimA == broadcasted_shape[idx]) ? strideA : 0;
        stridesB[idx] = (dimB == broadcasted_shape[idx]) ? strideB : 0;

        strideA *= (dimA == broadcasted_shape[idx]) ? dimA : 1;
        strideB *= (dimB == broadcasted_shape[idx]) ? dimB : 1;
    }

    for (int i = 0; i < broadcasted_size; i++) {
        int idx_result = i;
        int idxA = 0;
        int idxB = 0;
        for (int j = ndim-1; j >= 0; j--) {
            int pos = idx_result % broadcasted_shape[j];
            idx_result /= broadcasted_shape[j];
            idxA += stridesA[j] * pos;
            idxB += stridesB[j] * pos;
        }
        result_data[i] = tensorA->data[idxA] * tensorB->data[idxB];
    }

    free(stridesA);
    free(stridesB);
}

void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int N = tensorA->shape[0];

    for (int sum_idx = 0; sum_idx < N; sum_idx++) {
        result_data[0] += tensorA->data[sum_idx] * tensorB->data[sum_idx];
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

    for (int batch_idx = 0; batch_idx < total_num_matrices; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
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

void div_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] / tensorB->data[idx];
    }
}

void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] / divisor;
    }
}

void div_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* stridesA = (int*)malloc(ndim * sizeof(int));
    if (!stridesA) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int* stridesB = (int*)malloc(ndim * sizeof(int));
    if (!stridesB) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int strideA = 1, strideB = 1;
    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        stridesA[idx] = (dimA == broadcasted_shape[idx]) ? strideA : 0;
        stridesB[idx] = (dimB == broadcasted_shape[idx]) ? strideB : 0;

        strideA *= (dimA == broadcasted_shape[idx]) ? dimA : 1;
        strideB *= (dimB == broadcasted_shape[idx]) ? dimB : 1;
    }

    for (int i = 0; i < broadcasted_size; i++) {
        int idx_result = i;
        int idxA = 0;
        int idxB = 0;
        for (int j = ndim-1; j >= 0; j--) {
            int pos = idx_result % broadcasted_shape[j];
            idx_result /= broadcasted_shape[j];
            idxA += stridesA[j] * pos;
            idxB += stridesB[j] * pos;  
        }
        result_data[i] = tensorA->data[idxA] / tensorB->data[idxB];
    }

    free(stridesA);
    free(stridesB);
}

void flatten_cpu(Tensor* tensor, double* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        int flat_idx = i;
        int idx = 0;
        for (int j = tensor->ndim-1; j >= 0; j--) {
            int pos = flat_idx % tensor->shape[j];
            flat_idx /= tensor->shape[j];
            idx += tensor->strides[j] * pos;
        }
        result_data[i] = tensor->data[idx];
    }
}

void sum_cpu(Tensor* tensor, double* result_data) {
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[0] += tensor->data[idx];
    }
}

void sum_axis_cpu(Tensor* tensor, double* result_data, const int axis) {
    int size = tensor->size / tensor->shape[axis];

    int outer_size = 1;
    for (int idx = 0; idx < axis; idx++) {
        outer_size *= tensor->shape[idx];
    }

    int inner_size = size / outer_size;

    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            double sum = 0;
            for (int k = 0; k < tensor->shape[axis]; k++) {
                sum += tensor->data[inner_size * k + j + tensor->shape[axis] * inner_size * i];
            }
        result_data[j + inner_size * i] = sum;   
        }
    }
}

void mean_cpu(Tensor* tensor, double* result_data) {
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[0] += tensor->data[idx];
    }
    result_data[0] /= tensor->size;
}

void mean_axis_cpu(Tensor* tensor, double* result_data, const int axis) {
    sum_axis_cpu(tensor, result_data, axis);
    const int result_size = tensor->size / tensor->shape[axis];
    for (int idx = 0; idx < result_size; idx++) {
        result_data[idx] /= tensor->shape[axis];
    }
}

void pow_cpu(Tensor* tensor, double exponent, double* result_data) {
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[idx] = pow(tensor->data[idx], exponent);
    }
}

void ones_cpu(double* result_data, int size) {
    for (int idx = 0; idx < size; idx++) {
        result_data[idx] = 1.0;
    }
}

void outer_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int i = 0; i < tensorA->size; i++) {
        for (int j = 0; j < tensorB->size; j++) {
            result_data[tensorB->size * i + j] = tensorA->data[i] * tensorB->data[j];
        }
    }
}