#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstddef>

#include "jetdl/linalg/kernel.h"

namespace {

cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "CUDA: failed to create cuBLAS handle\n");
    }
  }
  return handle;
}

}  // namespace

// C = A @ B (NoTrans, NoTrans)
// Device-pointer kernel: all pointers (d_a, d_b, d_c) must already be on GPU
void matmul_kernel_cuda(const float* d_a, const float* d_b, float* d_c,
                        const size_t M, const size_t K, const size_t N,
                        const size_t lda, const size_t ldb, const size_t ldc) {
  cublasHandle_t handle = get_cublas_handle();
  const float alpha = 1.0f, beta = 0.0f;

  // Row-major C = A @ B computed as column-major C' = B' @ A'
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
              d_b, (int)ldb, d_a, (int)lda, &beta, d_c, (int)ldc);
}

// C = A @ B^T (NoTrans, Trans)
// A is (M, K), B is (N, K), C is (M, N)
void matmul_nt_kernel_cuda(const float* d_a, const float* d_b, float* d_c,
                           const size_t M, const size_t K, const size_t N,
                           const size_t lda, const size_t ldb,
                           const size_t ldc) {
  cublasHandle_t handle = get_cublas_handle();
  const float alpha = 1.0f, beta = 0.0f;

  // Row-major C = A @ B^T computed as column-major C' = B'^T @ A'
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
              d_b, (int)ldb, d_a, (int)lda, &beta, d_c, (int)ldc);
}

// C = A^T @ B (Trans, NoTrans)
// A is (K, M), B is (K, N), C is (M, N)
void matmul_tn_kernel_cuda(const float* d_a, const float* d_b, float* d_c,
                           const size_t M, const size_t K, const size_t N,
                           const size_t lda, const size_t ldb,
                           const size_t ldc) {
  cublasHandle_t handle = get_cublas_handle();
  const float alpha = 1.0f, beta = 0.0f;

  // Row-major C = A^T @ B computed as column-major C' = B' @ A'^T
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)K, &alpha,
              d_b, (int)ldb, d_a, (int)lda, &beta, d_c, (int)ldc);
}

// Outer product: C = x ⊗ y (rank-1 update)
// x is (M,), y is (N,), C is (M, N) row-major
void outer_product_kernel_cuda(const float* d_x, const float* d_y, float* d_c,
                               const size_t M, const size_t N) {
  cublasHandle_t handle = get_cublas_handle();
  const float alpha = 1.0f;

  // Zero out C first
  cudaMemset(d_c, 0, M * N * sizeof(float));

  // Row-major outer product x ⊗ y where result is (M, N)
  // In column-major (cuBLAS view): we want C[i,j] = x[i] * y[j]
  // cublasSger computes A = alpha * x * y^T + A (column-major)
  // For row-major (M, N): column-major sees (N, M), so we swap and compute
  // y ⊗ x in col-major which gives us x ⊗ y in row-major
  cublasSger(handle, (int)N, (int)M, &alpha, d_y, 1, d_x, 1, d_c, (int)N);
}

// Matrix-vector multiply: y = A @ x (NoTrans)
// A is (M, N), x is (N,), y is (M,)
void matvec_kernel_cuda(const float* d_a, const float* d_x, float* d_y,
                        const size_t M, const size_t N, const size_t lda) {
  cublasHandle_t handle = get_cublas_handle();
  const float alpha = 1.0f, beta = 0.0f;

  // Row-major A @ x where A is (M, N) and x is (N,)
  // Column-major sees A as (N, M), so A in col-major is Trans
  // y = A'^T @ x in col-major (where A' is col-major view of row-major A)
  cublasSgemv(handle, CUBLAS_OP_N, (int)N, (int)M, &alpha, d_a, (int)lda, d_x,
              1, &beta, d_y, 1);
}

// Transposed matrix-vector multiply: y = A^T @ x
// A is (M, N), x is (M,), y is (N,)
void matvec_t_kernel_cuda(const float* d_a, const float* d_x, float* d_y,
                          const size_t M, const size_t N, const size_t lda) {
  cublasHandle_t handle = get_cublas_handle();
  const float alpha = 1.0f, beta = 0.0f;

  // Row-major A^T @ x where A is (M, N) and x is (M,)
  // Column-major sees A as (N, M), so A^T in col-major is NoTrans
  // y = A' @ x in col-major (where A' is col-major view of row-major A)
  cublasSgemv(handle, CUBLAS_OP_T, (int)N, (int)M, &alpha, d_a, (int)lda, d_x,
              1, &beta, d_y, 1);
}
