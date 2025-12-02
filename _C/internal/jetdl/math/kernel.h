#ifndef MATH_KERNEL_H
#define MATH_KERNEL_H

#include <cstddef>

void c_add_kernel(const float* a, const float* b, float* c, const size_t N);

void c_add_a_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_add_b_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_add_scalars_kernel(const float* a, const float* b, float* c);

void c_sub_kernel(const float* a, const float* b, float* c, const size_t N);

void c_sub_a_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_sub_b_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_sub_scalars_kernel(const float* a, const float* b, float* c);

void c_mul_kernel(const float* a, const float* b, float* c, const size_t N);

void c_mul_a_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_mul_b_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_mul_scalars_kernel(const float* a, const float* b, float* c);

void c_div_kernel(const float* a, const float* b, float* c, const size_t N);

void c_div_a_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_div_b_scalar_kernel(const float* a, const float* b, float* c,
                           const size_t N);

void c_div_scalars_kernel(const float* a, const float* b, float* c);

void c_pow_kernel(float* dest, const float src, const int k);

void c_total_sum_kernel(float* dest, const float* src, const size_t N);

void c_sum_over_axes_kernel(float* dest, const float* src,
                            const size_t* dest_idxs, const size_t N);

void c_total_mean_kernel(float* dest, const float* src, const size_t N);

void c_mean_over_axes_kernel(float* dest, const float* src,
                             const size_t* dest_idxs, const size_t divisor,
                             const size_t N);

// CPU

void c_add_cpu(const float* a, const float* b, float* c, const size_t N);

void c_add_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_add_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_add_scalars_cpu(const float* a, const float* b, float* c);

void c_sub_cpu(const float* a, const float* b, float* c, const size_t N);

void c_sub_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_sub_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_sub_scalars_cpu(const float* a, const float* b, float* c);

void c_mul_cpu(const float* a, const float* b, float* c, const size_t N);

void c_mul_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_mul_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_mul_scalars_cpu(const float* a, const float* b, float* c);

void c_div_cpu(const float* a, const float* b, float* c, const size_t N);

void c_div_a_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_div_b_scalar_cpu(const float* a, const float* b, float* c,
                        const size_t N);

void c_div_scalars_cpu(const float* a, const float* b, float* c);

void c_pow_cpu(float* dest, const float src, const int k);

void c_total_sum_cpu(float* dest, const float* src, const size_t N);

void c_sum_over_axes_cpu(float* dest, const float* src, const size_t* dest_idxs,
                         const size_t N);

void c_total_mean_cpu(float* dest, const float* src, const size_t N);

void c_mean_over_axes_cpu(float* dest, const float* src,
                          const size_t* dest_idxs, const size_t divisor,
                          const size_t N);

// CUDA

void c_add_cuda(const float* a, const float* b, float* c, const size_t N);

void c_add_a_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_add_b_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_add_scalars_cuda(const float* a, const float* b, float* c);

void c_sub_cuda(const float* a, const float* b, float* c, const size_t N);

void c_sub_a_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_sub_b_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_sub_scalars_cuda(const float* a, const float* b, float* c);

void c_mul_cuda(const float* a, const float* b, float* c, const size_t N);

void c_mul_a_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_mul_b_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_mul_scalars_cuda(const float* a, const float* b, float* c);

void c_div_cuda(const float* a, const float* b, float* c, const size_t N);

void c_div_a_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_div_b_scalar_cuda(const float* a, const float* b, float* c,
                         const size_t N);

void c_div_scalars_cuda(const float* a, const float* b, float* c);

void c_pow_cuda(float* dest, const float src, const int k);

void c_total_sum_cuda(float* dest, const float* src, const size_t N);

void c_sum_over_axes_cuda(float* dest, const float* src,
                          const size_t* dest_idxs, const size_t N);

void c_total_mean_cuda(float* dest, const float* src, const size_t N);

void c_mean_over_axes_cuda(float* dest, const float* src,
                           const size_t* dest_idxs, const size_t divisor,
                           const size_t N);

#endif
