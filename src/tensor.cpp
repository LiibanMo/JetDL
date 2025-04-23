#include <iostream>
#include <cstring>
#include "tensor.h"
#include "cpu.h"

Tensor* create_tensor(double* data, int* shape, int ndim) {

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    tensor->ndim = ndim;

    tensor->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!tensor->shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    memcpy(tensor->shape, shape, tensor->ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    tensor->strides[ndim-1] = 1;
    for (int idx = ndim-2; idx >= 0; idx--) {
        tensor->strides[idx] = tensor->strides[idx+1] * tensor->shape[idx+1];
    }

    tensor->size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        tensor->size *= tensor->shape[idx];
    }

    tensor->data = (double*)malloc(tensor->size * sizeof(double));
    if (!tensor->data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    memcpy(tensor->data, data, tensor->size * sizeof(double));

    return tensor;
}

double get_item(Tensor* tensor, int* indices) {
    int index = 0;
    for (int idx = 0; idx < tensor->ndim; idx++) {
        index += tensor->strides[idx] * indices[idx];
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

Tensor* add_broadcasted(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* broadcasted_shape = (int*)malloc(ndim * sizeof(int));
    if (!broadcasted_shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        broadcasted_shape[idx] = (dimA > dimB) ? dimA : dimB;
    }

    int broadcasted_size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        broadcasted_size *= broadcasted_shape[idx];
    }

    double* result_data = (double*)malloc(broadcasted_size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    add_broadcasted_cpu(tensorA, tensorB, result_data, broadcasted_shape, broadcasted_size);

    return create_tensor(result_data, broadcasted_shape, ndim);
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

Tensor* sub_broadcasted(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* broadcasted_shape = (int*)malloc(ndim * sizeof(int));
    if (!broadcasted_shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        broadcasted_shape[idx] = (dimA > dimB) ? dimA : dimB; 
    }

    int broadcasted_size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        broadcasted_size *= broadcasted_shape[idx];
    }

    double* result_data = (double*)malloc(broadcasted_size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    sub_broadcasted_cpu(tensorA, tensorB, result_data, broadcasted_shape, broadcasted_size);

    return create_tensor(result_data, broadcasted_shape, ndim);
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

Tensor* mul_broadcasted(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* broadcasted_shape = (int*)malloc(ndim * sizeof(int));
    if (!broadcasted_shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        broadcasted_shape[idx] = (dimA > dimB) ? dimA : dimB;
    }

    int broadcasted_size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        broadcasted_size *= broadcasted_shape[idx];
    }

    double* result_data = (double*)malloc(broadcasted_size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    mul_broadcasted_cpu(tensorA, tensorB, result_data, broadcasted_shape, broadcasted_size);

    return create_tensor(result_data, broadcasted_shape, ndim);
}

Tensor* vector_dot_product(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = 1;

    int* shape = (int*)malloc(sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = 1;

    double* result_data = (double*)malloc(sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    vector_dot_product_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* matmul_2d_2d(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = 2;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    shape[0] = tensorA->shape[0];
    shape[1] = tensorB->shape[1];

    const int size = shape[0] * shape[1];
    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    matmul_2d_2d_cpu(tensorA, tensorB, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* matmul_broadcasted(Tensor* tensorA, Tensor* tensorB) {
    if (tensorA->ndim >= tensorB->ndim) {

        Tensor* view_tensorB = (Tensor*)malloc(sizeof(Tensor));
        if (!view_tensorB) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        view_tensorB->data = tensorB->data;

        view_tensorB->ndim = tensorA->ndim;

        view_tensorB->shape = (int*)malloc(view_tensorB->ndim * sizeof(int));
        if (!view_tensorB->shape) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }

        view_tensorB->strides = (int*)malloc(view_tensorB->ndim * sizeof(int));
        if (!view_tensorB->strides) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        
        for (int idx = 0; idx < view_tensorB->ndim; idx++) {
            if (idx < view_tensorB->ndim - tensorB->ndim) {
                view_tensorB->shape[idx] = 1;
                view_tensorB->strides[idx] = 0;
            } else {
                view_tensorB->shape[idx] = tensorB->shape[idx - (view_tensorB->ndim - tensorB->ndim)];
                view_tensorB->strides[idx] = tensorB->strides[idx - (view_tensorB->ndim - tensorB->ndim)];
            }
        }

        int size = 1;
        for (int idx = 0; idx < tensorA->ndim-1; idx++) {
            size *= tensorA->shape[idx];
        }
        size *= tensorB->shape[tensorB->ndim-1];

        double* result_data = (double*)malloc(size * sizeof(double));
        if (!result_data) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }

        char broadcasted[] = "B";
        matmul_broadcasted_cpu(tensorA, view_tensorB, result_data, broadcasted);

        free_tensor(view_tensorB);
        
        const int ndim = tensorA->ndim;

        int* shape = (int*)malloc(ndim * sizeof(int));
        if (!shape) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        for (int idx = 0; idx < ndim-1; idx++) {
            shape[idx] = tensorA->shape[idx];
        }
        shape[ndim-1] = tensorB->shape[tensorB->ndim-1];

        return create_tensor(result_data, shape, ndim);
        
    } else if (tensorA->ndim < tensorB->ndim) {

        Tensor* view_tensorA = (Tensor*)malloc(sizeof(Tensor));
        if (!view_tensorA) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        view_tensorA->data = tensorA->data;

        view_tensorA->ndim = tensorB->ndim;

        view_tensorA->shape = (int*)malloc(view_tensorA->ndim * sizeof(int));
        if (!view_tensorA->shape) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }

        view_tensorA->strides = (int*)malloc(view_tensorA->ndim * sizeof(int));
        if (!view_tensorA->strides) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }

        for (int idx = 0; idx < view_tensorA->ndim; idx++) {
            if (idx < view_tensorA->ndim - tensorA->ndim) {
                view_tensorA->shape[idx] = 1;
                view_tensorA->strides[idx] = 0;
            } else {
                view_tensorA->shape[idx] = tensorA->shape[idx - (view_tensorA->ndim - tensorA->ndim)];
                view_tensorA->strides[idx] = tensorA->strides[idx - (view_tensorA->ndim - tensorA->ndim)];
            }
        }

        int size = 1;
        for (int idx = 0; idx < tensorB->ndim-2; idx++) {
            size *= tensorB->shape[idx];
        }
        if (tensorA->ndim != 1) {
            size *= tensorA->shape[tensorA->ndim-2];
        }
        size *= tensorB->shape[tensorB->ndim-1];

        double* result_data = (double*)malloc(size * sizeof(double));
        if (!result_data) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }

        char broadcasted[] = "A";
        matmul_broadcasted_cpu(view_tensorA, tensorB, result_data, broadcasted);

        free_tensor(view_tensorA);

        int ndim = tensorB->ndim;
        if (tensorA->ndim == 1) {
            ndim--;
        }

        int* shape = (int*)malloc(ndim * sizeof(int));
        if (!shape) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        switch (ndim) {
            case 2 : {
                shape[ndim-2] = tensorB->shape[tensorB->ndim-3];
                shape[ndim-1] = tensorB->shape[tensorB->ndim-1];
                break;
            }
            default : {
                for (int idx = 0; idx < ndim-2; idx++) {
                    shape[idx] = tensorB->shape[idx];
                }
                shape[ndim-2] = tensorA->shape[tensorA->ndim-2];
                shape[ndim-1] = tensorB->shape[tensorB->ndim-1];
                break;
            }
        }

        return create_tensor(result_data, shape, ndim);

    } else {
        fprintf(stderr, "Wrong implementation of matmul used. This function is for: matmtul with operations with either one being N-D where N>2.\n");
        return NULL;
    }
}

Tensor* div_tensor(Tensor* tensorA, Tensor* tensorB) {
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

    div_tensor_cpu(tensorA, tensorB, result_data);

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

Tensor* div_broadcasted(Tensor* tensorA, Tensor* tensorB) {
    const int ndim = (tensorA->ndim > tensorB->ndim) ? tensorA->ndim : tensorB->ndim;

    int* broadcasted_shape = (int*)malloc(ndim * sizeof(int));
    if (!broadcasted_shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int idx = ndim-1; idx >= 0; idx--) {
        int idxA = tensorA->ndim - ndim + idx;
        int idxB = tensorB->ndim - ndim + idx;

        int dimA = (idxA >= 0) ? tensorA->shape[idxA] : 1;
        int dimB = (idxB >= 0) ? tensorB->shape[idxB] : 1;

        broadcasted_shape[idx] = (dimA > dimB) ? dimA : dimB;
    }

    int broadcasted_size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        broadcasted_size *= broadcasted_shape[idx];
    }

    double* result_data = (double*)malloc(broadcasted_size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    div_broadcasted_cpu(tensorA, tensorB, result_data, broadcasted_shape, broadcasted_size);

    return create_tensor(result_data, broadcasted_shape, ndim);
}

Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {
    const int ndim = new_ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    for (int idx = 0; idx < ndim; idx++) {
        shape[idx] = new_shape[idx];
    }

    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    assign_tensor_data_cpu(tensor, result_data);

    return create_tensor(result_data, shape, ndim);

}

Tensor* flatten_tensor(Tensor* tensor) {
    const int ndim = 1;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    shape[0] = tensor->size;

    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    flatten_cpu(tensor, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* transpose_tensor(Tensor* tensor) {
    int* shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int idx = 0; idx < tensor->ndim; idx++) {
        shape[idx] = tensor->shape[tensor->ndim-1-idx];
    }

    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    assign_tensor_data_cpu(tensor, result_data);

    Tensor* result_tensor =  create_tensor(result_data, shape, tensor->ndim);

    for (int idx = 0; idx < tensor->ndim; idx++) {
        result_tensor->strides[idx] = tensor->strides[tensor->ndim-1-idx];
    }

    return result_tensor;
}

Tensor* matrix_transpose_tensor(Tensor* tensor) {
    int* shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    if (tensor->ndim > 2) {
        for (int idx = 0; idx < tensor->ndim-2; idx++) {
            shape[idx] = tensor->shape[idx];
        }
    }
    shape[tensor->ndim-2] = tensor->shape[tensor->ndim-1];
    shape[tensor->ndim-1] = tensor->shape[tensor->ndim-2];

    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    assign_tensor_data_cpu(tensor, result_data);

    Tensor* result_tensor = create_tensor(result_data, shape, tensor->ndim);

    result_tensor->strides[tensor->ndim-1] = result_tensor->shape[tensor->ndim-2];
    result_tensor->strides[tensor->ndim-2] = 1;

    return result_tensor;
}

Tensor* sum_tensor(Tensor* tensor) {
    const int ndim = 0;

    int* shape = (int*)malloc(sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    shape[0] = 0;

    double* result_data = (double*)malloc(sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
 
    sum_cpu(tensor, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* sum_axis_tensor(Tensor* tensor, const int axis) {
    const int ndim = tensor->ndim - 1;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int idx = 0; idx < tensor->ndim; idx++) {
        int idx_shape = (idx <= axis) ? idx : idx - 1;
        shape[idx_shape] = tensor->shape[idx];
    }

    const int size = tensor->size / tensor->shape[axis];

    double* result_data = (double*)malloc(size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    sum_axis_cpu(tensor, result_data, axis);

    return create_tensor(result_data, shape, ndim);
}

Tensor* mean_tensor(Tensor* tensor) {
    const int ndim = 0;

    int* shape = (int*)malloc(sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    shape[0] = 0;

    double* result_data = (double*)malloc(sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    mean_cpu(tensor, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* mean_axis_tensor(Tensor* tensor, const int axis) {
    const int ndim = tensor->ndim - 1;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    for (int idx = 0; idx < tensor->ndim; idx++) {
        int idx_shape = (idx <= axis) ? idx : idx - 1;
        shape[idx_shape] = tensor->shape[idx];
    }

    const int size = tensor->size / tensor->shape[axis];

    double* result_data = (double*)malloc(size * sizeof(int));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    mean_axis_cpu(tensor, result_data, axis);

    return create_tensor(result_data, shape, ndim);
}

Tensor* pow_tensor(Tensor* tensor, double exponent) {
    double* result_data = (double*)malloc(tensor->size * sizeof(double));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    pow_cpu(tensor, exponent, result_data);

    return create_tensor(result_data, tensor->shape, tensor->ndim);
}

void free_tensor(Tensor* tensor_ptr) {
    if (tensor_ptr) {
        free(tensor_ptr);
        tensor_ptr = NULL;
    }
}