#include <iostream>
#include <cmath>
#include <omp.h>
#include "string.h"
#include "lib.h"
#include <cblas.h>

void assign_tensor_data_cpu(Tensor* tensor, float* result_data) {
    #pragma omp parallel for
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = tensor->data[i];
    }
}

void make_contiguous_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        int lin_idx = i;
        int idx = 0;
        for (int j = tensor->ndim-1; j >= 0; j--) {
            int pos_idx = lin_idx % tensor->shape[j];
            lin_idx /= tensor->shape[j];
            idx += pos_idx * tensor->strides[j];
        }
        result_data[i] = tensor->data[idx];
    }
}

void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data) {
    cblas_scopy(tensorA->size, tensorA->data, 1, result_data, 1);
    cblas_saxpy(tensorA->size, 1.0f, tensorB->data, 1, result_data, 1);
}

void scalar_add_tensor_cpu(Tensor* tensorA, float operand, float* result_data) {
    #pragma omp parallel for
    for (int i = 0; i < tensorA->size; i++) {
        result_data[i] = tensorA->data[i] + operand;
    }
}

void add_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape, int broadcasted_size) {
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

    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(broadcasted_size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < broadcasted_size; i++) {
        int idx_result = i;
        int idxA = 0;
        int idxB = 0;
        #pragma omp parallel for reduction(+:idxA, idxB)
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

void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data) {
    cblas_scopy(tensorA->size, tensorA->data, 1, result_data, 1);
    cblas_saxpy(tensorA->size, -1.0f, tensorB->data, 1, result_data, 1);
}

void scalar_sub_tensor_cpu(Tensor* tensorA, float operand, float* result_data) {
    #pragma omp parallel for
    for (int i = 0; i < tensorA->size; i++) {
        result_data[i] = tensorA->data[i] - operand;
    }
}

void sub_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= broadcasted_shape[i];
    }

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

    #pragma omp parallel for
    for (int i = 0; i < total_size; i++) {
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

void scalar_mul_tensor_cpu(Tensor* tensorA, float operand, float* result_data) {
    cblas_scopy(tensorA->size, tensorA->data, 1, result_data, 1);
    cblas_sscal(tensorA->size, operand, result_data, 1);
}

void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data) {
    #pragma omp parallel for
    for (int i = 0; i < tensorA->size; i++) {
        result_data[i] = tensorA->data[i] * tensorB->data[i];
    }
}

void mul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape, int broadcasted_size) {
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

    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(broadcasted_size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
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

void div_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data) {
    #pragma omp parallel for
    for (int i = 0; i < tensorA->size; i++) {
        result_data[i] = tensorA->data[i] / tensorB->data[i];
    }
}

void scalar_div_tensor_cpu(Tensor* tensorA, float divisor, float* result_data) {
    cblas_scopy(tensorA->size, tensorA->data, 1, result_data, 1);
    cblas_sscal(tensorA->size, 1.0f / divisor, result_data, 1);
}

void div_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape, int broadcasted_size) {
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

    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(broadcasted_size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
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

void flatten_cpu(Tensor* tensor, float* result_data) {
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

void sum_cpu(Tensor* tensor, float* result_data) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < tensor->size; i++) {
        sum += tensor->data[i];
    }
    result_data[0] = sum;
}

void sum_axis_cpu(Tensor* tensor, float* result_data, const int axis) {
    int size = tensor->size / tensor->shape[axis];
    int outer_size = 1;
    for (int idx = 0; idx < axis; idx++) {
        outer_size *= tensor->shape[idx];
    }
    int inner_size = size / outer_size;

    #pragma omp parallel for
    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < tensor->shape[axis]; k++) {
                int idx = i * tensor->shape[axis] * inner_size + k * inner_size + j;
                sum += tensor->data[idx];
            }
            result_data[i * inner_size + j] = sum;
        }
    }
}

void mean_cpu(Tensor* tensor, float* result_data) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < tensor->size; i++) {
        sum += tensor->data[i];
    }
    result_data[0] = sum / tensor->size;
}

void mean_axis_cpu(Tensor* tensor, float* result_data, const int axis) {
    sum_axis_cpu(tensor, result_data, axis);
    const int result_size = tensor->size / tensor->shape[axis];
    cblas_sscal(result_size, 1.0f / tensor->shape[axis], result_data, 1);
}

void pow_cpu(Tensor* tensor, float exponent, float* result_data) {
    #pragma omp parallel for
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = std::pow(tensor->data[i], exponent);
    }
}