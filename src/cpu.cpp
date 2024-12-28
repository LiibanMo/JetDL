#include "tensor.h"
#include "cpu_utils.h"

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

void matmul_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int ndimA = tensorA->ndim;
    const int ndimB = tensorB->ndim;

    if (ndimA == 1 && ndimB == 1) {
        vector_matmul_vector(tensorA, tensorB, result_data);
    } else if (ndimA == 1 && ndimB == 2) {
        vector_matmul_matrix(tensorA, tensorB, result_data);
    } else if (ndimA == 1 && ndimB == 3) {
        vector_matmul_tensor(tensorA, tensorB, result_data);
    } else if (ndimA == 2 && ndimB == 1) {
        matrix_matmul_vector(tensorA, tensorB, result_data);
    } else if (ndimA == 2 && ndimB == 2) {
        matrix_matmul_matrix(tensorA, tensorB, result_data);
    } else if (ndimA == 2 && ndimB == 3) {
        matrix_matmul_tensor(tensorA, tensorB, result_data);
    } else if (ndimA == 3 && ndimB == 1) {
        tensor_matmul_vector(tensorA, tensorB, result_data);
    } else if (ndimA == 3 && ndimB == 2) {
        tensor_matmul_matrix(tensorA, tensorB, result_data);
    } else if (ndimA == 3 && ndimB == 3) {
        tensor_matmul_tensor(tensorA, tensorB, result_data);
    }
}

void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] / divisor;
    }
}