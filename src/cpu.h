#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data);
void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data);
void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, float* result_data);

#endif