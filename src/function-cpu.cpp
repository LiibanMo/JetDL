#include <iostream>
#include <cmath>
#include <omp.h>
#include "string.h"
#include "lib.h"


void ones_cpu(float* result_data, int size) {
    #pragma omp parallel for
    for (int idx = 0; idx < size; idx++) {
        result_data[idx] = 1.0f;
    }
}

void exp_cpu(Tensor* tensorA, float* result_data) {
    #pragma omp parallel for
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = std::exp(tensorA->data[idx]);
    }
}

void log_cpu(Tensor* tensor, float* result_data) {
    #pragma omp parallel for
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[idx] = std::log(tensor->data[idx]);
    }
}