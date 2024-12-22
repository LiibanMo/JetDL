#include <iostream>
#include "tensor.h"
#include "cpu.h"

Tensor* create_tensor(double* data, int* shape, int ndim) {

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

double get_item(Tensor* tensor, int* indices) {
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        index += tensor->strides[i] * indices[i];
    }
    return tensor->data[index];
}

Tensor* add_tensor(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = tensorA->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = tensorA->shape[idx];
    }

    double* result_data = (double*)malloc(tensorA->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    add_tensor_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* scalar_add_tensor(Tensor* tensor, double operand) {
    const int ndim = tensor->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = tensor->shape[idx];
    }

    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    scalar_add_tensor_cpu(tensor, operand, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* sub_tensor(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = tensorA->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
       shape[idx] = tensorA->shape[idx]; 
    }  

    double* result_data = (double*)malloc(tensorA->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    subtract_tensor_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* scalar_sub_tensor(Tensor* tensorA, double operand) {
    const int ndim = tensorA->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = tensorA->shape[idx];
    }

    double* result_data = (double*)malloc(tensorA->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    scalar_sub_tensor_cpu(tensorA, operand, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* hadamard_mul_tensor(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = tensorA->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = tensorA->shape[idx];
    }

    double* result_data = (double*)malloc(tensorA->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    hadamard_mul_tensor_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* inner_product_tensor(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = 1;

    int* shape = (int*)malloc(1);
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = 1;

    double* result_data = (double*)malloc(1);
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    inner_product_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* matmul_tensor_vector(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = tensorA->ndim;

    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = M;

    double* result_data = (double*)malloc(M * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_tensor_vector_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* matmul_tensor(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = tensorA->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    const int size = tensorA->shape[1] * tensorB->shape[0];

    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_tensor_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* batch_matmul_tensor(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = tensorA->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    shape[0] = tensorA->shape[0];
    shape[1] = tensorA->shape[1];
    shape[2] = tensorB->shape[2];

    int size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        size *= shape[idx];
    }

    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    batch_matmul_tensor_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* scalar_mul_tensor(Tensor* tensorA, double operand) {
    const int ndim = tensorA->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = tensorA->shape[idx];
    }

    double* result_data = (double*)malloc(tensorA->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    scalar_mul_tensor_cpu(tensorA, operand, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* scalar_div_tensor(Tensor* tensor, double divisor) {
    const int ndim = tensor->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = tensor->shape[idx];
    }

    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    scalar_div_tensor_cpu(tensor, divisor, result_data);

    return create_tensor(result_data, shape, ndim);
}

void free_tensor(Tensor* tensor) {
    if (tensor) {
        free(tensor);
        tensor = NULL;
    } 
}

void free_data(Tensor* tensor) {
    if (tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
    }
}

void free_shape(Tensor* tensor) {
    if (tensor->shape) {
        free(tensor->shape);
        tensor->shape = NULL;
    }
}

void free_strides(Tensor* tensor) {
    if (tensor->strides) {
        free(tensor->strides);
        tensor->strides = NULL;
    }
}
