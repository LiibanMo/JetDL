#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstddef>

#include "jetdl/linalg/kernel.h"

// Device-pointer kernel: all pointers (d_a, d_b, d_c) must already be on GPU
void matmul_kernel_cuda(const float* d_a, const float* d_b, float* d_c,
                        const size_t M, const size_t K, const size_t N,
                        const size_t lda, const size_t ldb, const size_t ldc) {
  // (M, K) @ (K, N) = (M , N)
  cublasHandle_t handle;
  cublasStatus_t status;

  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA: (../linalg/product/kernel.cu) status unsuccessful.");
    return;
  }

  const float alpha = 1.0f, beta = 0.0f;

  /*
  cublasSgemm assumes input to be column-major
  C = A @ B => C^T = B^T @ A^T
  B^T: shape, strides = (N, K), (1, N)
  A^T.shape, strides = (K, M), (1, K)
  C^T.shape, strides = (N, M), (1, N)
  */
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
              d_b, (int)ldb, d_a, (int)lda, &beta, d_c, (int)ldc);

  cublasDestroy(handle);
}
