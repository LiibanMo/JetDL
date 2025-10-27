#include "jetdl/math/kernel.h"

#include <cstddef>

#if defined(__ARM_NEON__) || defined(__AVX2__)

#include "jetdl/simd.h"

#if defined(__ARM_NEON__)
constexpr size_t BLOCK_SIZE = 4;
#elif defined(__AVX2__)
constexpr size_t BLOCK_SIZE = 8;
#endif

void c_total_sum_cpu(float* dest, const float* src, const size_t N) {
  JETDL_FLOAT32_V partial_sums = JETDL_DUP_n_F32_V(0.0f);
  size_t i = 0;
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    const JETDL_FLOAT32_V src_vec = JETDL_LD1_F32_V(src + i);
    partial_sums = JETDL_ADD_F32_V(partial_sums, src_vec);
  }
  *dest += JETDL_SUM_F32_V(partial_sums);

  for (; i < N; i++) {
    *dest += src[i];
  }
}

void c_sum_over_axes_cpu(float* dest, const float* src, const size_t* dest_idxs,
                         const size_t N) {
  for (size_t i = 0; i < N; i++) {
    dest[dest_idxs[i]] += src[i];
  }
}

void c_total_mean_cpu(float* dest, const float* src, const size_t N) {
  c_total_sum_cpu(dest, src, N);
  *dest /= N;
}

void c_mean_over_axes_cpu(float* dest, const float* src,
                          const size_t* dest_idxs, const size_t divisor,
                          const size_t N) {
  for (size_t i = 0; i < N; i++) {
    dest[dest_idxs[i]] += src[i] / divisor;
  }
}

#else

void c_total_sum_cpu(float* dest, const float* src, const size_t N) {
  for (size_t i = 0; i < N; i++) {
    *dest += src[i];
  }
}

void c_sum_over_axes_cpu(float* dest, const float* src, const size_t* dest_idxs,
                         const size_t N) {
  for (size_t i = 0; i < N; i++) {
    dest[dest_idxs[i]] += src[i];
  }
}

void c_total_mean_cpu(float* dest, const float* src, const size_t N) {
  for (size_t i = 0; i < N; i++) {
    *dest += src[i] / N;
  }
}

void c_mean_over_axes_cpu(float* dest, const float* src,
                          const size_t* dest_idxs, const size_t divisor,
                          const size_t N) {
  for (size_t i = 0; i < N; i++) {
    dest[dest_idxs[i]] += src[i] / divisor;
  }
}

#endif
