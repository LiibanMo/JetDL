#include "jetdl/math/kernel.h"

#include <cstddef>

#if defined(__ARM_NEON__) || defined(__AVX2__)

#include "jetdl/simd.h"

constexpr size_t BLOCK_SIZE = SIMD_SIZE * 2;
constexpr size_t N_VECS = BLOCK_SIZE / SIMD_SIZE;

void c_add_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V a_vec = JETDL_DUP_n_F32_V(*a);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_ADD_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = *a + b[i];
  }
}

void c_add_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V b_vec = JETDL_DUP_n_F32_V(*b);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_ADD_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = a[i] + *b;
  }
}

void c_add_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a + *b;
}

void c_sub_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V a_vec = JETDL_DUP_n_F32_V(*a);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_SUB_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = *a - b[i];
  }
}

void c_sub_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V b_vec = JETDL_DUP_n_F32_V(*b);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    c[i] = a[i] + *b;
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_SUB_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = a[i] - *b;
  }
}

void c_sub_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a - *b;
}

void c_mul_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V a_vec = JETDL_DUP_n_F32_V(*a);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_MUL_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = *a * b[i];
  }
}

void c_mul_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V b_vec = JETDL_DUP_n_F32_V(*b);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    c[i] = a[i] + *b;
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_MUL_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = a[i] * *b;
  }
}

void c_mul_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a * *b;
}

void c_div_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V a_vec = JETDL_DUP_n_F32_V(*a);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_DIV_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = *a / b[i];
  }
}

void c_div_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V b_vec = JETDL_DUP_n_F32_V(*b);
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    c[i] = a[i] + *b;
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_DIV_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = a[i] / *b;
  }
}

void c_div_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a / *b;
}

void c_add_cpu(const float* a, const float* b, float* c, const size_t N) {
  size_t i = 0;
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_ADD_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

void c_add_inplace_cpu(float* a, const float* b, const size_t N) {
  size_t i = 0;
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(a + i + (SIMD_SIZE * j), JETDL_ADD_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    a[i] += b[i];
  }
}

void c_sub_cpu(const float* a, const float* b, float* c, const size_t N) {
  size_t i = 0;
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_SUB_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = a[i] - b[i];
  }
}

void c_mul_cpu(const float* a, const float* b, float* c, const size_t N) {
  size_t i = 0;
  for (; i + BLOCK_SIZE - 1 < N; i += BLOCK_SIZE) {
    for (size_t j = 0; j < N_VECS; j++) {
      const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(a + i + (SIMD_SIZE * j));
      const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(b + i + (SIMD_SIZE * j));
      JETDL_ST1_F32_V(c + i + (SIMD_SIZE * j), JETDL_MUL_F32_V(a_vec, b_vec));
    }
  }
  for (; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}

void c_div_cpu(const float* a, const float* b, float* c, const size_t N) {
  size_t i = 0;
  for (; i + SIMD_SIZE - 1 < N; i += SIMD_SIZE) {
    const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(&a[i]);
    const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(&b[i]);
    JETDL_ST1_F32_V(&c[i], JETDL_DIV_F32_V(a_vec, b_vec));
  }
  for (; i < N; i++) {
    c[i] = a[i] / b[i];
  }
}

// Fused backward for division: result = -a * grad / (b * b)
void c_div_backward_b_cpu(const float* a, const float* b, const float* grad,
                          float* result, const size_t N) {
  size_t i = 0;
  const JETDL_FLOAT32_V neg_one = JETDL_DUP_n_F32_V(-1.0f);
  for (; i + SIMD_SIZE - 1 < N; i += SIMD_SIZE) {
    const JETDL_FLOAT32_V a_vec = JETDL_LD1_F32_V(&a[i]);
    const JETDL_FLOAT32_V b_vec = JETDL_LD1_F32_V(&b[i]);
    const JETDL_FLOAT32_V grad_vec = JETDL_LD1_F32_V(&grad[i]);
    // Compute: -a * grad / (b * b)
    const JETDL_FLOAT32_V b_sq = JETDL_MUL_F32_V(b_vec, b_vec);
    const JETDL_FLOAT32_V numer = JETDL_MUL_F32_V(JETDL_MUL_F32_V(neg_one, a_vec), grad_vec);
    JETDL_ST1_F32_V(&result[i], JETDL_DIV_F32_V(numer, b_sq));
  }
  for (; i < N; i++) {
    result[i] = -a[i] * grad[i] / (b[i] * b[i]);
  }
}

#else

void c_add_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = *a + b[i];
  }
}

void c_add_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = a[i] + *b;
  }
}

void c_add_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a + *b;
}

void c_sub_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = *a - b[i];
  }
}

void c_sub_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = a[i] - *b;
  }
}

void c_sub_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a - *b;
}

void c_mul_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = a[i] * *b;
  }
}

void c_mul_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = *a * b[i];
  }
}

void c_mul_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a * *b;
}

void c_div_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = *a / b[i];
  }
}

void c_div_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = a[i] / *b;
  }
}

void c_div_scalars_cpu(const float* a, const float* b, float* c) {
  *c = *a / *b;
}

void c_add_cpu(const float* a, const float* b, float* c, const size_t N) {
  for (size_t i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

void c_add_inplace_cpu(float* a, const float* b, const size_t N) {
  for (size_t i = 0; i < N; i++) {
    a[i] += b[i];
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

// Fused backward for division: result = -a * grad / (b * b)
void c_div_backward_b_cpu(const float* a, const float* b, const float* grad,
                          float* result, const size_t N) {
  for (size_t i = 0; i < N; i++) {
    result[i] = -a[i] * grad[i] / (b[i] * b[i]);
  }
}

#endif
