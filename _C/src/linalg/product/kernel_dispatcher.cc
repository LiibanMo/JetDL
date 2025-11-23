#include "jetdl/linalg/kernel.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/utils/device.h"
#endif

void matmul_kernel(const float* a, const float* b, float* c, const size_t M,
                   const size_t K, const size_t N, const size_t lda,
                   const size_t ldb, const size_t ldc) {
#ifdef JETDL_WITH_CUDA
  if (is_cuda_available()) {
    matmul_kernel_cuda(a, b, c, M, K, N, lda, ldb, ldc);
  } else {
    matmul_kernel_cpu(a, b, c, M, K, N, lda, ldb, ldc);
  }
#else
  matmul_kernel_cpu(a, b, c, M, K, N, lda, ldb, ldc);
#endif
}
