#ifndef JETDL_LINALG_KERNEL_H
#define JETDL_LINALG_KERNEL_H

#define BLOCK_N_ROWS 6
#define BLOCK_N_COLS 8

#include <cstddef>

// NOTE: Device-aware dispatch is handled in matmul.cc directly.
// Call matmul_kernel_cpu for CPU tensors, matmul_kernel_cuda for CUDA tensors.

// C = A @ B (NoTrans, NoTrans)
// A is (M, K), B is (K, N), C is (M, N)
void matmul_kernel_cpu(const float* a, const float* b, float* c, const size_t M,
                       const size_t K, const size_t N, const size_t lda,
                       const size_t ldb, const size_t ldc);

void matmul_kernel_cuda(const float* a, const float* b, float* c,
                        const size_t M, const size_t K, const size_t N,
                        const size_t lda, const size_t ldb, const size_t ldc);

// C = A @ B^T (NoTrans, Trans) - used for dA = dC @ B^T in backward
// A is (M, K), B is (N, K), C is (M, N)
void matmul_nt_kernel_cpu(const float* a, const float* b, float* c,
                          const size_t M, const size_t K, const size_t N,
                          const size_t lda, const size_t ldb, const size_t ldc);

void matmul_nt_kernel_cuda(const float* a, const float* b, float* c,
                           const size_t M, const size_t K, const size_t N,
                           const size_t lda, const size_t ldb, const size_t ldc);

// C = A^T @ B (Trans, NoTrans) - used for dB = A^T @ dC in backward
// A is (K, M), B is (K, N), C is (M, N)
void matmul_tn_kernel_cpu(const float* a, const float* b, float* c,
                          const size_t M, const size_t K, const size_t N,
                          const size_t lda, const size_t ldb, const size_t ldc);

void matmul_tn_kernel_cuda(const float* a, const float* b, float* c,
                           const size_t M, const size_t K, const size_t N,
                           const size_t lda, const size_t ldb, const size_t ldc);

// Outer product: C = x ⊗ y, where x is (M,), y is (N,), C is (M, N)
// Used for dA in MatVecBackward: dA = grad ⊗ x
void outer_product_kernel_cpu(const float* x, const float* y, float* c,
                              const size_t M, const size_t N);

void outer_product_kernel_cuda(const float* x, const float* y, float* c,
                               const size_t M, const size_t N);

// Matrix-vector multiply: y = A @ x (NoTrans)
// A is (M, N), x is (N,), y is (M,)
// Used for dA in VecMatBackward: dA = B @ grad
void matvec_kernel_cpu(const float* a, const float* x, float* y,
                       const size_t M, const size_t N, const size_t lda);

void matvec_kernel_cuda(const float* a, const float* x, float* y,
                        const size_t M, const size_t N, const size_t lda);

// Transposed matrix-vector multiply: y = A^T @ x
// A is (M, N), x is (M,), y is (N,)
// Used for dB in MatVecBackward: dB = A^T @ grad
void matvec_t_kernel_cpu(const float* a, const float* x, float* y,
                         const size_t M, const size_t N, const size_t lda);

void matvec_t_kernel_cuda(const float* a, const float* x, float* y,
                          const size_t M, const size_t N, const size_t lda);

#endif
