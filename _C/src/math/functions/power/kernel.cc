#include "jetdl/math/kernel.h"

#ifdef __ARM_NEON__

void c_pow_cpu(float* dest, const float src, const int k) {
  size_t n = k;
  float r = src;
  *dest = 1.0f;

  while (n > 0) {
    if (n & 1) {
      *dest *= r;
    }
    r *= r;
    n >>= 1;
  }
}

#endif
