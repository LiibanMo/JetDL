#ifndef FUNCTION_H
#define FUNCTION_H

#include "lib.h"

void ones_cpu(float* result_data, int size);
void exp_cpu(Tensor* tensorA, float* result_data);
void log_cpu(Tensor* tensor, float* result_data);


#endif