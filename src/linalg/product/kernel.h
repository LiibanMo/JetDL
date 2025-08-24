#ifndef LINALG_PRODUCT_KERNEL_H
#define LINALG_PRODUCT_KERNEL_H

#define BLOCK_N_ROWS 6
#define BLOCK_N_COLS 8

#ifdef __cplusplus
extern "C" {
#endif

    void c_matmul_cpu(float* a, float* b, float* c, const int x, const int y, const int l, const int r, const int p, const int n);

#ifdef __cplusplus
}
#endif

#endif