#include "jetdl/linalg/kernel.h"

#include <cstring>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// C = A @ B (NoTrans, NoTrans)
void matmul_kernel_cpu(const float* a, const float* b, float* c, const size_t M,
                       const size_t K, const size_t N, const size_t lda,
                       const size_t ldb, const size_t ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
              1.0f, a, (int)lda, b, (int)ldb, 0.0f, c, (int)ldc);
}

// C = A @ B^T (NoTrans, Trans)
// A is (M, K), B is (N, K), C is (M, N)
void matmul_nt_kernel_cpu(const float* a, const float* b, float* c,
                          const size_t M, const size_t K, const size_t N,
                          const size_t lda, const size_t ldb,
                          const size_t ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)M, (int)N, (int)K,
              1.0f, a, (int)lda, b, (int)ldb, 0.0f, c, (int)ldc);
}

// C = A^T @ B (Trans, NoTrans)
// A is (K, M), B is (K, N), C is (M, N)
void matmul_tn_kernel_cpu(const float* a, const float* b, float* c,
                          const size_t M, const size_t K, const size_t N,
                          const size_t lda, const size_t ldb,
                          const size_t ldc) {
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)M, (int)N, (int)K,
              1.0f, a, (int)lda, b, (int)ldb, 0.0f, c, (int)ldc);
}

// Outer product: C = x ⊗ y (rank-1 update)
// x is (M,), y is (N,), C is (M, N) row-major
void outer_product_kernel_cpu(const float* x, const float* y, float* c,
                              const size_t M, const size_t N) {
  // Zero out C first, then use sger for rank-1 update: C = alpha * x * y^T + C
  // For row-major: use RowMajor and swap to get x * y^T effect
  std::memset(c, 0, M * N * sizeof(float));
  cblas_sger(CblasRowMajor, (int)M, (int)N, 1.0f, x, 1, y, 1, c, (int)N);
}

// Matrix-vector multiply: y = A @ x (NoTrans)
// A is (M, N), x is (N,), y is (M,)
void matvec_kernel_cpu(const float* a, const float* x, float* y,
                       const size_t M, const size_t N, const size_t lda) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)M, (int)N, 1.0f, a, (int)lda, x,
              1, 0.0f, y, 1);
}

// Transposed matrix-vector multiply: y = A^T @ x
// A is (M, N), x is (M,), y is (N,)
void matvec_t_kernel_cpu(const float* a, const float* x, float* y,
                         const size_t M, const size_t N, const size_t lda) {
  cblas_sgemv(CblasRowMajor, CblasTrans, (int)M, (int)N, 1.0f, a, (int)lda, x,
              1, 0.0f, y, 1);
}