#include <iostream>
#include <cmath>
#include <omp.h>
#include "string.h"
#include "tensor.h"


void assign_tensor_data_cpu(Tensor* tensor, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensor->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[idx] = tensor->data[idx];
    }
}

void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] + tensorB->data[idx];
    }
}

void scalar_add_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
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

void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] - tensorB->data[idx]; 
    }
}

void scalar_sub_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
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
        result_data[i] = tensorA->data[idxA] - tensorB->data[idxB];
    }

    free(stridesA);
    free(stridesB);
}   

void scalar_mul_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] * operand;
    }
}

void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
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
        result_data[i] = tensorA->data[idxA] * tensorB->data[idxB];
    }

    free(stridesA);
    free(stridesB);
}

void div_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] / tensorB->data[idx];
    }
}

void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensorA->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
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
        result_data[i] = tensorA->data[idxA] / tensorB->data[idxB];
    }

    free(stridesA);
    free(stridesB);
}

void flatten_cpu(Tensor* tensor, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensor->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
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
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensor->size / 10, 1));

    double sum = 0;

    #pragma omp parallel for reduction(+:sum) num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensor->size; idx++) {
        sum += tensor->data[idx];
    }
    result_data[0] = sum;
}

void sum_axis_cpu(Tensor* tensor, double* result_data, const int axis) {
    int size = tensor->size / tensor->shape[axis];

    int outer_size = 1;
    for (int idx = 0; idx < axis; idx++) {
        outer_size *= tensor->shape[idx];
    }

    int inner_size = size / outer_size;

    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(outer_size * inner_size / 10, 1));

    #pragma omp parallel for, collapse(2), num_threads(NUM_THREADS)
    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            double sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int k = 0; k < tensor->shape[axis]; k++) {
                sum += tensor->data[inner_size * k + j + tensor->shape[axis] * inner_size * i];
            }
        result_data[j + inner_size * i] = sum;   
        }
    }
}

void mean_cpu(Tensor* tensor, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensor->size / 10, 1));

    double sum = 0;

    #pragma omp parallel for reduction(+:sum) num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensor->size; idx++) {
        sum += tensor->data[idx];
    }
    result_data[0] = sum;
    result_data[0] /= tensor->size;
}

void mean_axis_cpu(Tensor* tensor, double* result_data, const int axis) {
    sum_axis_cpu(tensor, result_data, axis);
    const int result_size = tensor->size / tensor->shape[axis];

    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(result_size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < result_size; idx++) {
        result_data[idx] /= tensor->shape[axis];
    }
}

void pow_cpu(Tensor* tensor, double exponent, double* result_data) {
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(tensor->size / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[idx] = pow(tensor->data[idx], exponent);
    }
}