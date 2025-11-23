#include "jetdl/linalg/kernel.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

void matmul_kernel_cpu(const float* a, const float* b, float* c, const size_t M,
                       const size_t K, const size_t N, const size_t lda,
                       const size_t ldb, const size_t ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
              1.0f, a, (int)lda, b, (int)ldb, 0.0f, c, (int)ldc);
}