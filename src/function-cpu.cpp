#include <iostream>
#include <cmath>
#include <omp.h>
#include "string.h"
#include "lib.h"


void ones_cpu(float* result_data, int size) {
    const int NUM_ITERS = size;
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(NUM_ITERS / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < size; idx++) {
        result_data[idx] = 1.0;
    }
}

void exp_cpu(Tensor* tensorA, float* result_data) {
    const int NUM_ITERS = tensorA->size;
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(NUM_ITERS / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = std::exp(tensorA->data[idx]);
    }
}

void log_cpu(Tensor* tensor, float* result_data) {
    const int NUM_ITERS = tensor->size;
    const int NUM_THREADS = std::max(omp_get_max_threads(), std::min(NUM_ITERS / 10, 1));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[idx] = std::log(tensor->data[idx]);
    }
}