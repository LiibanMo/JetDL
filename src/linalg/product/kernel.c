#include "kernel.h"

#if defined(__ARM_NEON__)

#include <arm_neon.h>

void c_matmul_cpu(float* a, float* b, float* c, const size_t x, const size_t y, const size_t p, const size_t n) {
    // (m, n) @ (n, p) = (m, p)
    float32x4_t t00 = vdupq_n_f32(0.0f);
    float32x4_t t01 = vdupq_n_f32(0.0f);

    float32x4_t t10 = vdupq_n_f32(0.0f);
    float32x4_t t11 = vdupq_n_f32(0.0f);

    float32x4_t t20 = vdupq_n_f32(0.0f);
    float32x4_t t21 = vdupq_n_f32(0.0f);

    float32x4_t t30 = vdupq_n_f32(0.0f);
    float32x4_t t31 = vdupq_n_f32(0.0f);

    float32x4_t t40 = vdupq_n_f32(0.0f);
    float32x4_t t41 = vdupq_n_f32(0.0f);

    float32x4_t t50 = vdupq_n_f32(0.0f);
    float32x4_t t51 = vdupq_n_f32(0.0f);

    for (int k = 0; k < n; k++) {
        float32x4_t b0 = vld1q_f32(&b[k * p + y]);
        float32x4_t b1 = vld1q_f32(&b[k * p + y + 4]);

        float32x4_t a0 = vdupq_n_f32(a[x * n + k]);
        t00 = vfmaq_f32(t00, a0, b0);
        t01 = vfmaq_f32(t01, a0, b1);

        float32x4_t a1 = vdupq_n_f32(a[(x + 1) * n + k]);
        t10 = vfmaq_f32(t10, a1, b0);
        t11 = vfmaq_f32(t11, a1, b1);

        float32x4_t a2 = vdupq_n_f32(a[(x + 2) * n + k]);
        t20 = vfmaq_f32(t20, a2, b0);
        t21 = vfmaq_f32(t21, a2, b1);

        float32x4_t a3 = vdupq_n_f32(a[(x + 3) * n + k]);
        t30 = vfmaq_f32(t30, a3, b0);
        t31 = vfmaq_f32(t31, a3, b1);

        float32x4_t a4 = vdupq_n_f32(a[(x + 4) * n + k]);
        t40 = vfmaq_f32(t40, a4, b0);
        t41 = vfmaq_f32(t41, a4, b1);

        float32x4_t a5 = vdupq_n_f32(a[(x + 5) * n + k]);
        t50 = vfmaq_f32(t50, a5, b0);
        t51 = vfmaq_f32(t51, a5, b1);
    }

    vst1q_f32(&c[x * p + y], t00);
    vst1q_f32(&c[x * p + y + 4], t01);

    vst1q_f32(&c[(x + 1) * p + y], t10);
    vst1q_f32(&c[(x + 1) * p + y + 4], t11);

    vst1q_f32(&c[(x + 2) * p + y], t20);
    vst1q_f32(&c[(x + 2) * p + y + 4], t21);

    vst1q_f32(&c[(x + 3) * p + y], t30);
    vst1q_f32(&c[(x + 3) * p + y + 4], t31);

    vst1q_f32(&c[(x + 4) * p + y], t40);
    vst1q_f32(&c[(x + 4) * p + y + 4], t41);

    vst1q_f32(&c[(x + 5) * p + y], t50);
    vst1q_f32(&c[(x + 5) * p + y + 4], t51);
}

#else
#include <cblas.h>

void c_matmul_cpu(
    float* a, float* b, float* c, const size_t x, const size_t y, const size_t p, const size_t n
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    const int k = r - l;    
    
    cblas_sgemm(
        CblasRowMajor,    
        CblasNoTrans,     
        CblasNoTrans,    
        BLOCK_N_ROWS,               
        BLOCK_N_COLS,               
        k,               
        alpha,           
        &a[x * n + l],   
        n,               
        &b[l * p + y],   
        p,               
        beta,            
        &c[x * p + y],   
        p               
    );
}

#endif
