#ifndef OPS_CPU_H
#define OPS_CPU_H

#include "lib.h"

void assign_tensor_data_cpu(Tensor* tensor, float* result_data);
void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data);
void scalar_add_tensor_cpu(Tensor* tensorA, float operand, float* result_data);
void add_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape, int broadcasted_size);
void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data);
void scalar_sub_tensor_cpu(Tensor* tensorA, float operand, float* result_data);
void sub_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape, int broadcasted_size);
void scalar_mul_tensor_cpu(Tensor* tensorA, float operand, float* result_data);
void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data);
void mul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape, int broadcasted_size);
void div_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data);
void scalar_div_tensor_cpu(Tensor* tensorA, float divisor, float* result_data);
void div_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data, int* broadcasted_shape, int broadcasted_size);
void flatten_cpu(Tensor* tensor, float* result_data);
void sum_cpu(Tensor* tensor, float* result_data);
void sum_axis_cpu(Tensor* tensor, float* result_data, const int axis);
void mean_cpu(Tensor* tensor, float* result_data);
void mean_axis_cpu(Tensor* tensor, float* result_data, const int axis);
void pow_cpu(Tensor* tensor, float exponent, float* result_data);

#endif