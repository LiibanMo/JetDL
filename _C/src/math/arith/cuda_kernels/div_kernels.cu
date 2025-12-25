#include <cuda_runtime.h>

#include "jetdl/math/kernel.h"

constexpr int NUM_THREADS_PER_BLOCK = 256;

__global__ void c_div_cuda_kernel(const float* a, const float* b, float* c,
                                  const size_t N) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    c[i] = a[i] / b[i];
  }
}

__global__ void c_div_a_scalar_cuda_kernel(const float* a, const float* b,
                                           float* c, const size_t N) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    c[i] = a[0] / b[i];
  }
}

__global__ void c_div_b_scalar_cuda_kernel(const float* a, const float* b,
                                           float* c, const size_t N) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    c[i] = a[i] / b[0];
  }
}

// Device-pointer kernels: all pointers must already be on GPU
void c_div_cuda(const float* d_a, const float* d_b, float* d_c,
                const size_t N) {
  const int NUM_BLOCKS_PER_GRID =
      (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  c_div_cuda_kernel<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK>>>(d_a, d_b,
                                                                     d_c, N);
}

void c_div_a_scalar_cuda(const float* d_a, const float* d_b, float* d_c,
                         const size_t N) {
  const int NUM_BLOCKS_PER_GRID =
      (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  c_div_a_scalar_cuda_kernel<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK>>>(
      d_a, d_b, d_c, N);
}

void c_div_b_scalar_cuda(const float* d_a, const float* d_b, float* d_c,
                         const size_t N) {
  const int NUM_BLOCKS_PER_GRID =
      (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  c_div_b_scalar_cuda_kernel<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK>>>(
      d_a, d_b, d_c, N);
}

void c_div_scalars_cuda(const float* d_a, const float* d_b, float* d_c) {
  // For scalar division, use a single thread
  c_div_cuda_kernel<<<1, 1>>>(d_a, d_b, d_c, 1);
}