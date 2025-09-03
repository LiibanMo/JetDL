#ifndef MATH_FUNCTION_KERNEL_H
#define MATH_FUNCTION_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

    void c_total_sum_cpu(float* dest, const float* src, const size_t N);
    void c_sum_over_axes_cpu(float* dest, const float* src, const size_t* dest_idxs, const size_t N);

#ifdef __cplusplus
}
#endif

#endif