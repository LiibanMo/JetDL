#ifndef LINALG_CPU_H
#define LINALG_CPU_H

#include "tensor.h"

void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matmul_2d_2d_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);
void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, char broadcasted[]);
void outer_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data);

#endif