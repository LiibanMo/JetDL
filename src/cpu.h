#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void scalar_add_tensor_cpu(Tensor* tensorA, double operand, double* result_data);
void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void scalar_sub_tensor_cpu(Tensor* tensorA, double operand, double* result_data);
void scalar_mul_tensor_cpu(Tensor* tensorA, double operand, double* result_data);
void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matmul_2d_2d_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, char broadcasted[]);
void matmul_broadcasted_1D_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, char broadcasted[]);
void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data);
#endif