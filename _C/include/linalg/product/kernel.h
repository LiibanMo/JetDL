#ifndef LINALG_PRODUCT_KERNEL_H
#define LINALG_PRODUCT_KERNEL_H

#define BLOCK_N_ROWS 6
#define BLOCK_N_COLS 8

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

    void c_matmul_cpu(float* a, float* b, float* c, const size_t x, const size_t y, const size_t p, const size_t n);

#ifdef __cplusplus
}
#endif

#endif