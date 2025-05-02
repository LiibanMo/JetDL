#include <iostream>
#include "tensor.h"
#include "cpu.h"
#include "routines.h"

Tensor* ones(int* shape, const int ndim) {
    int size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        size *= shape[idx];
    }

    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    ones_cpu(result_data, size);

    Tensor* result_tensor = create_tensor(result_data, shape, ndim);

    free(result_data);

    return result_tensor;
}

Tensor* outer(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = 2;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    shape[0] = tensorA->size;
    shape[1] = tensorB->size;

    const int size = shape[0] * shape[1];

    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    outer_cpu(tensorA, tensorB, result_data);

    Tensor* result_tensor = create_tensor(result_data, shape, ndim);

    free(shape);
    free(result_data);

    return result_tensor;
}

Tensor* c_exp(Tensor* tensor) {
    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    exp_cpu(tensor, result_data);

    Tensor* result_tensor = create_tensor(result_data, tensor->shape, tensor->ndim);

    free(result_data);

    return result_tensor;
}

Tensor* c_log(Tensor* tensor) {
    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    log_cpu(tensor, result_data);

    Tensor* result_tensor = create_tensor(result_data, tensor->shape, tensor->ndim);

    free(result_data);

    return result_tensor;
}