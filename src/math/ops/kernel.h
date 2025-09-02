#ifndef MATH_OPS_KERNEL_H
#define MATH_OPS_KERNEL_H

#define BLOCK_N_COLS 8

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

    void c_add_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_add_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_add_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_add_scalars_cpu(const float* a, const float* b, float* c);
    
    void c_sub_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_sub_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_sub_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_sub_scalars_cpu(const float* a, const float* b, float* c);
    
    void c_mul_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_mul_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_mul_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_mul_scalars_cpu(const float* a, const float* b, float* c);
    
    void c_div_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_div_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_div_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N);
    void c_div_scalars_cpu(const float* a, const float* b, float* c);

#ifdef __cplusplus
}
#endif

#endif