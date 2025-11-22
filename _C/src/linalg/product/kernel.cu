#include <cstddef>

#include "jetdl/linalg/kernel.h"

#if defined(__CUDACC__)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

void matmul_kernel(const float* a, const float* b, float* c, const size_t M,
                   const size_t K, const size_t N, const size_t lda,
                   const size_t ldb, const size_t ldc) {
  // (M, K) @ (K, N) = (M , N)
#if defined(__CUDACC__)
  cublasHandle_t handle;
  cublasStatus_t status;

  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA: (../linalg/product/kernel.cu) status unsuccessful.");
    return;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, M * K * sizeof(float));
  cudaMalloc(&d_b, K * N * sizeof(float));
  cudaMalloc(&d_c, M * N * sizeof(float));

  cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * N * sizeof(float), cudaMemcpyHostToDevice);

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

  cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cublasDestroy(handle);

#else
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
              1.0f, a, (int)lda, b, (int)ldb, 0.0f, c, (int)ldc);
#endif
}
