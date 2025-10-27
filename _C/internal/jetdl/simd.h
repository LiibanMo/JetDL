#ifndef JETDL_SIMD_H
#define JETDL_SIMD_H

#include <cstddef>

#if defined(__ARM_NEON__)

#include <arm_neon.h>
// ARM NEON uses 128-bit (4 floats) quad-word vectors (float32x4_t)
constexpr size_t SIMD_SIZE = 4;

// NEON dtypes
#define JETDL_FLOAT32_V float32x4_t

// NEON functions
#define JETDL_LD1_F32_V vld1q_f32
#define JETDL_ST1_F32_V vst1q_f32
#define JETDL_DUP_n_F32_V vdupq_n_f32
#define JETDL_ADD_F32_V vaddq_f32
#define JETDL_SUB_F32_V vsubq_f32
#define JETDL_MUL_F32_V vmulq_f32
#define JETDL_DIV_F32_V vdivq_f32
#define JETDL_SUM_F32_V vaddvq_f32

#elif defined(__AVX2__)

#include <immintrin.h>
// AVX2 uses 256-bit (8 floats) YMM registers (__m256)
constexpr size_t SIMD_SIZE = 8;

// AVX2 dtypes
#define JETDL_FLOAT32_V __m256

// AVX2 functions
#define JETDL_LD1_F32_V _mm256_loadu_ps
#define JETDL_ST1_F32_V _mm256_storeu_ps
#define JETDL_DUP_n_F32_V _mm256_set1_ps
#define JETDL_ADD_F32_V _mm256_add_ps
#define JETDL_SUB_F32_V _mm256_sub_ps
#define JETDL_MUL_F32_V _mm256_mul_ps
#define JETDL_DIV_F32_V _mm256_div_ps
#define JETDL_SUM_F32_V(vec)                                    \
  ({                                                            \
    __m128 sum_low = _mm_add_ps(_mm256_castps256_ps128(vec),    \
                                _mm256_extractf128_ps(vec, 1)); \
    __m128 sum_high = _mm_hadd_ps(sum_low, sum_low);            \
    __m128 result = _mm_hadd_ps(sum_high, sum_high);            \
    _mm_cvtss_f32(result);                                      \
  })

#endif

#endif