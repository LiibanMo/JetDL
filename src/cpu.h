#ifndef CPU_H
#define CPU_H

#include "tensor.h"

// METHODS
void assign_tensor_data_cpu(Tensor* tensor, double* result_data);
void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void scalar_add_tensor_cpu(Tensor* tensorA, double operand, double* result_data);
void add_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size);
void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void scalar_sub_tensor_cpu(Tensor* tensorA, double operand, double* result_data);
void sub_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size);
void scalar_mul_tensor_cpu(Tensor* tensorA, double operand, double* result_data);
void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void mul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size);
void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matmul_2d_2d_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, char broadcasted[]);
void div_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data);
void div_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, int* broadcasted_shape, int broadcasted_size);
void flatten_cpu(Tensor* tensor, double* result_data);
void sum_cpu(Tensor* tensor, double* result_data);
void sum_axis_cpu(Tensor* tensor, double* result_data, const int axis);
void mean_cpu(Tensor* tensor, double* result_data);
void mean_axis_cpu(Tensor* tensor, double* result_data, const int axis);
void pow_cpu(Tensor* tensor, double exponent, double* result_data);
// FUNCTIONS
void ones_cpu(double* result_data, int size);
void outer_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void exp_cpu(Tensor* tensorA, double* result_data);

#endif