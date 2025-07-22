#include "kernel.hpp"

#if defined(__ARM_NEON__)

#include <arm_neon.h>

void c_add_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += 4) {
        __builtin_prefetch(&a[i+4]);
        __builtin_prefetch(&b[i+4]);

        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t r0 = vaddq_f32(a0, b0);
        
        vst1q_f32(&c[i], r0);
    }
}

void c_sub_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += 4) {
        __builtin_prefetch(&a[i+4]);
        __builtin_prefetch(&b[i+4]);

        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t r0 = vsubq_f32(a0, b0);
        
        vst1q_f32(&c[i], r0);
    }
}

void c_mul_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += 4) {
        __builtin_prefetch(&a[i+4]);
        __builtin_prefetch(&b[i+4]);

        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t r0 = vmulq_f32(a0, b0);
        
        vst1q_f32(&c[i], r0);
    }
}

void c_div_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += 4) {
        __builtin_prefetch(&a[i+4]);
        __builtin_prefetch(&b[i+4]);

        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t r0 = vdivq_f32(a0, b0);
        
        vst1q_f32(&c[i], r0);
    }
}

#else

void c_add_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void c_sub_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] - b[i];
    }
}

void c_mul_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
}

void c_div_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] / b[i];
    }
}

#endif