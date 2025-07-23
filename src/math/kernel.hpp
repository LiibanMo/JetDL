#pragma once

#define BLOCK_N_COLS 8

void c_add_cpu(const float* a, const float* b, float* c, const int N);
void c_sub_cpu(const float* a, const float* b, float* c, const int N);
void c_mul_cpu(const float* a, const float* b, float* c, const int N);
void c_div_cpu(const float* a, const float* b, float* c, const int N);