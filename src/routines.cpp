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
        return NULL;
    }

    ones_cpu(result_data, size);

    return create_tensor(result_data, shape, ndim);
}

Tensor* outer(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = 2;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    shape[0] = tensorA->size;
    shape[1] = tensorB->size;

    const int size = shape[0] * shape[1];

    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    outer_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}