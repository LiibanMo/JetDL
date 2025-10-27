#ifndef JETDL_LINALG_KERNEL_H
#define JETDL_LINALG_KERNEL_H

#define BLOCK_N_ROWS 6
#define BLOCK_N_COLS 8

#include <cstddef>

void gemm_packed_kernel(const float* a, const float* b, float* c,
                        const size_t m_block, const size_t k_block,
                        const size_t n_block, const size_t N_orig);

void matmul_cpu(const float* a, const float* b, float* c, const size_t M,
                const size_t K, const size_t N, const size_t lda,
                const size_t ldb, const size_t ldc);

#endif
