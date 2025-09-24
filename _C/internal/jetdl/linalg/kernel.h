#ifndef JETDL_LINALG_KERNEL_H
#define JETDL_LINALG_KERNEL_H

#define BLOCK_N_ROWS 6
#define BLOCK_N_COLS 8

#include <cstddef>

void c_matmul_cpu(const float* a, const float* b, float* c, const size_t x,
                  const size_t y, const size_t p, const size_t n);

#endif
