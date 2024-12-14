#include <iostream>
#include "tensor.h"
#include "cpu.h"

Tensor* create_tensor(float* data, int* shape, int ndim) {

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    tensor->strides[ndim-1] = 1;
    for (int idx = ndim-2; idx >= 0; idx--) {
        tensor->strides[idx] = tensor->strides[idx+1]*tensor->shape[idx+1];
    }

    tensor->size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        tensor->size *= tensor->shape[idx];
    }

    return tensor;
}

float get_item(Tensor* tensor, int* indices) {
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        index += tensor->strides[i] * indices[i];
    }
    return tensor->data[index];
}

Tensor* add_tensor(Tensor* tensorA, Tensor* tensorB) {
    if (tensorA->ndim != tensorB->ndim) {
        fprintf(stderr, "Inconsistent number of dimensions for tensor-tensor addition.\n");
        return NULL;
    }
    const int ndim = tensorA->ndim;

    for (int idx = 0; idx < ndim; idx++) {
        if (tensorA->shape[idx] != tensorB->shape[idx]) {
            fprintf(stderr, "Inconsistent dimensions in shape for tensor-tensor addition.\n");
            return NULL;
        }
    }
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = tensorA->shape[idx];
    }

    float* result_data = (float*)malloc(tensorA->size * sizeof(float));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    add_tensor_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* subtract_tensor(Tensor* tensorA, Tensor* tensorB) {
    if (tensorA->ndim != tensorB->ndim) {
        fprintf(stderr, "Inconsistent number of dimensions for tensor-tensor subtraction.\n");
        return NULL;
    }
    const int ndim = tensorA->ndim;

    for (int idx = 0; idx < ndim; idx++) {
        if (tensorA->shape[idx] != tensorB->shape[idx]) {
            fprintf(stderr, "Inconsistent dimensions in shape for tensor-tensor subtraction.\n");
            return NULL;
        }
    }
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
       shape[idx] = tensorA->shape[idx]; 
    }  

    float* result_data = (float*)malloc(tensorA->size * sizeof(float));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    subtract_tensor_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}