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
Tensor* vector_matmul_vector(Tensor* tensorA, Tensor* tensorB) {
    // (N) @ (N) = (1)
    const int ndim = tensorA->ndim; // = 1

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = 1;

    double* result_data = (double*)malloc(shape[0] * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* vector_matmul_matrix(Tensor* tensorA, Tensor* tensorB) {
    // (N) @ (N, P) = (P)
    const int ndim = tensorB->ndim; // = 1

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorB->shape[1];

    double* result_data = (double*)malloc(shape[0] * sizeof(int));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}   

Tensor* vector_matmul_batched_tensor(Tensor* tensorA, Tensor* tensorB) {
    // (N) @ (B, N, P) = (B, P)
    const int ndim = 2;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorB->shape[0];
    shape[1] = tensorB->shape[2];

    const int size = shape[0] * shape[1];
    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* matrix_matmul_vector(Tensor* tensorA, Tensor* tensorB) {
    // (M, N) @ (N) = (M)
    const int ndim = tensorB->ndim; // = 1

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorA->shape[0];

    double* result_data = (double*)malloc(shape[0] * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* matrix_matmul_matrix(Tensor* tensorA, Tensor* tensorB) {
    // (M, N) @ (N, P) = (M, P)
    const int ndim = tensorA->ndim; // = 2
    
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorA->shape[0];
    shape[1] = tensorB->shape[1];

    const int size = shape[0] * shape[1];
    double* result_data = (double*)malloc(shape[0] * shape[1] * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* matrix_matmul_batched_tensor(Tensor* tensorA, Tensor* tensorB) {
    // (M, N) @ (B, N, P) = (B, M, P)
    const int ndim = tensorA->ndim; // = 3

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorB->shape[0];
    shape[1] = tensorA->shape[0];
    shape[2] = tensorB->shape[2];

    const int size = shape[0] * shape[1] * shape[2];
    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* batched_tensor_matmul_vector(Tensor* tensorA, Tensor* tensorB) {
    // (B, M, N) @ (N) = (B, M)
    const int ndim = 2;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorA->shape[0];
    shape[1] = tensorA->shape[1];

    const int size = shape[0] * shape[1];
    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* batched_tensor_matmul_matrix(Tensor* tensorA, Tensor* tensorB) {
    // (B, M, N) @ (N, P) = (B, M, P)
    const int ndim = tensorA->ndim; // = 3

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorA->shape[0];
    shape[1] = tensorA->shape[1];
    shape[2] = tensorB->shape[1];

    const int size = shape[0] * shape[1] * shape[2];
    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}
    

Tensor* batched_tensor_matmul_batched_tensor(Tensor* tensorA, Tensor* tensorB) {
    // (B, M, N) @ (B, N, P) = (B, M, P)
    const int ndim = tensorA->ndim; // = 3

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorA->shape[0];
    shape[1] = tensorA->shape[1];
    shape[2] = tensorB->shape[2];

    const int size = shape[0] * shape[1] * shape[2];
    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    matmul_cpu(tensorA, tensorB, result_data);

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
