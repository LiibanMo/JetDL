#include "kernel.hpp"

#if defined(__ARM_NEON__)

#include <arm_neon.h>

void c_add_cpu(const float* a, const float* b, float* c, const int N) {
    int i = 0;
    for (; i + BLOCK_N_COLS <= N; i += BLOCK_N_COLS) {
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        vst1q_f32(&c[i], vaddq_f32(a0, b0));
    } 
    for (; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void c_add_cpu(const float* a, const float b, float* c, const int N) {
    for (int i = 0; i < N; i += BLOCK_N_COLS) {
        __builtin_prefetch(&a[i+BLOCK_N_COLS]);
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t a1 = vld1q_f32(&a[i+4]);
        const float32x4_t b0 = vdupq_n_f32(b);
        vst1q_f32(&c[i], vaddq_f32(a0, b0));
        vst1q_f32(&c[i+4], vaddq_f32(a1, b0));
    }
}

void c_sub_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += BLOCK_N_COLS) {
        __builtin_prefetch(&a[i+BLOCK_N_COLS]);
        __builtin_prefetch(&b[i+BLOCK_N_COLS]);
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t a1 = vld1q_f32(&a[i+4]);
        const float32x4_t b1 = vld1q_f32(&b[i+4]);
        vst1q_f32(&c[i], vsubq_f32(a0, b0));
        vst1q_f32(&c[i+4], vsubq_f32(a1, b1));
    }
}

void c_sub_cpu(const float* a, const float b, float* c, const int N) {
    for (int i = 0; i < N; i += BLOCK_N_COLS) {
        __builtin_prefetch(&a[i+BLOCK_N_COLS]);
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t a1 = vld1q_f32(&a[i+4]);
        const float32x4_t b0 = vdupq_n_f32(b);
        vst1q_f32(&c[i], vsubq_f32(a0, b0));
        vst1q_f32(&c[i+4], vsubq_f32(a1, b0));
    }
}

void c_sub_cpu(const float a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += BLOCK_N_COLS) {
        __builtin_prefetch(&b[i+BLOCK_N_COLS]);
        const float32x4_t a0 = vdupq_n_f32(a);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t b1 = vld1q_f32(&b[i+4]);
        vst1q_f32(&c[i], vsubq_f32(a0, b0));
        vst1q_f32(&c[i+4], vsubq_f32(a0, b1));
    }
}

void c_mul_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += BLOCK_N_COLS) {
        __builtin_prefetch(&a[i+BLOCK_N_COLS]);
        __builtin_prefetch(&b[i+BLOCK_N_COLS]);
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t a1 = vld1q_f32(&a[i+4]);
        const float32x4_t b1 = vld1q_f32(&b[i+4]);
        vst1q_f32(&c[i], vmulq_f32(a0, b0));
        vst1q_f32(&c[i+4], vmulq_f32(a1, b1));
    }
}


void c_div_cpu(const float* a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += BLOCK_N_COLS) {
        __builtin_prefetch(&a[i+BLOCK_N_COLS]);
        __builtin_prefetch(&b[i+BLOCK_N_COLS]);
        const float32x4_t a0 = vld1q_f32(&a[i]);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t a1 = vld1q_f32(&a[i+4]);
        const float32x4_t b1 = vld1q_f32(&b[i+4]);
        vst1q_f32(&c[i], vdivq_f32(a0, b0));
        vst1q_f32(&c[i+4], vdivq_f32(a1, b1));
    }
}

void c_div_cpu(const float a, const float* b, float* c, const int N) {
    for (int i = 0; i < N; i += BLOCK_N_COLS) {
        __builtin_prefetch(&b[i+BLOCK_N_COLS]);
        const float32x4_t a0 = vdupq_n_f32(a);
        const float32x4_t b0 = vld1q_f32(&b[i]);
        const float32x4_t b1 = vld1q_f32(&b[i+4]);
        vst1q_f32(&c[i], vdivq_f32(a0, b0));
        vst1q_f32(&c[i+4], vdivq_f32(a0, b1));
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