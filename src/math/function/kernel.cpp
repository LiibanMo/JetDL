#include "kernel.hpp"

void c_total_sum_cpu(const float* src, float* dest, const int size) {
    for (int i = 0; i < size; i++) {
        dest[0] += src[i];
    }
}

void c_sum_cpu(const float* src, float* dest, const int* dest_idxs, const int size) {
    for (int i = 0; i < size; i++) {
        dest[dest_idxs[i]] += src[i];
    }
}   