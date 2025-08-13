#pragma once

#define BLOCK_N_COLS 4

void c_total_sum_cpu(const float* src, float* dest, const int size);
void c_sum_cpu(const float* src, float* dest, const int* dest_idxs, const int size);