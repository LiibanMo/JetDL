#include "kernel.h"

void c_add_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = *a + b[i];
    }
}

void c_add_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] + *b;
    }
}

void c_add_scalars_cpu(const float* a, const float* b, float* c) {
    *c = *a + *b;
}

void c_sub_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = *a - b[i];
    }
}

void c_sub_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] - *b;
    }
}

void c_sub_scalars_cpu(const float* a, const float* b, float* c) {
    *c = *a - *b;
}

void c_mul_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] * *b;
    }
}

void c_mul_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = *a * b[i];
    }
}

void c_mul_scalars_cpu(const float* a, const float* b, float* c) {
    *c = *a * *b;
}

void c_div_a_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = *a / b[i];
    }
}

void c_div_b_scalar_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] / *b;
    }
}

void c_div_scalars_cpu(const float* a, const float* b, float* c) {
    *c = *a / *b;
}

#if defined(__ARM_NEON__)

#include <arm_neon.h>

void c_add_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i += BLOCK_N_COLS) {
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t a1 = vld1q_f32(&a[i+4]);
        const float32x4_t b1 = vld1q_f32(&b[i+4]);
        vst1q_f32(&c[i], vaddq_f32(a0, b0));
        vst1q_f32(&c[i+4], vaddq_f32(a1, b1));
    }
}

void c_sub_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i += BLOCK_N_COLS) {
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t a1 = vld1q_f32(&a[i + 4]);
        const float32x4_t b1 = vld1q_f32(&b[i + 4]);
        vst1q_f32(&c[i], vsubq_f32(a0, b0));
        vst1q_f32(&c[i + 4], vsubq_f32(a1, b1));
    }
}

void c_mul_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i += BLOCK_N_COLS) {
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t a1 = vld1q_f32(&a[i + 4]);
        const float32x4_t b1 = vld1q_f32(&b[i + 4]);
        vst1q_f32(&c[i], vmulq_f32(a0, b0));
        vst1q_f32(&c[i + 4], vmulq_f32(a1, b1));
    }
}

void c_div_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i += BLOCK_N_COLS) {
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t a1 = vld1q_f32(&a[i + 4]);
        const float32x4_t b1 = vld1q_f32(&b[i + 4]);
        vst1q_f32(&c[i], vdivq_f32(a0, b0));
        vst1q_f32(&c[i + 4], vdivq_f32(a1, b1));
    }
}

#else

void c_add_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void c_sub_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] - b[i];
    }
}

void c_mul_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
}

void c_div_cpu(const float* a, const float* b, float* c, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] / b[i];
    }
}

#endif