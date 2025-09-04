#include "jetdl/C/math/kernel.h"

void c_total_sum_cpu(float *dest, const float *src, const size_t N) {
  for (size_t i = 0; i < N; i++) {
    *dest += src[i];
  }
}

void c_sum_over_axes_cpu(float *dest, const float *src, const size_t *dest_idxs,
                         const size_t N) {
  for (size_t i = 0; i < N; i++) {
    dest[dest_idxs[i]] += src[i];
  }
}
