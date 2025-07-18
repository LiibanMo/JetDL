#pragma once

#define BLOCK_N_ROWS 6
#define BLOCK_N_COLS 8

void c_matmul_cpu(float* a, float* b, float* c, const int x, const int y, const int l, const int r, const int p, const int n);