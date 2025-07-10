#pragma once

#ifndef PRODUCT_CPU_HPP
#define PRODUCT_CPU_HPP

void c_dot_cpu(const float* data1, const float* data2, float* result_data, const int n);
void c_matmul_cpu(float* a, float* b, float* c, const int x, const int y, const int l, const int r, const int p, const int n);

#endif