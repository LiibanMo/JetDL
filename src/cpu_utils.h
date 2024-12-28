#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include "tensor.h"

void vector_matmul_vector(Tensor* tensorA, Tensor* tensorB, double* result_data);
void vector_matmul_matrix(Tensor* tensorA, Tensor* tensorB, double* result_data);
void vector_matmul_tensor(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matrix_matmul_vector(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matrix_matmul_matrix(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matrix_matmul_tensor(Tensor* tensorA, Tensor* tensorB, double* result_data);
void tensor_matmul_vector(Tensor* tensorA, Tensor* tensorB, double* result_data);
void tensor_matmul_matrix(Tensor* tensorA, Tensor* tensorB, double* result_data);
void tensor_matmul_tensor(Tensor* tensorA, Tensor* tensorB, double* result_data);

#endif