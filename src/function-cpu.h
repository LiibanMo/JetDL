#ifndef FUNCTION_H
#define FUNCTION_H

#include "tensor.h"

void ones_cpu(double* result_data, int size);
void exp_cpu(Tensor* tensorA, double* result_data);
void log_cpu(Tensor* tensor, double* result_data);


#endif