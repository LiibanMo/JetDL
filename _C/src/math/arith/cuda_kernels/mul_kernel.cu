#include <cuda_runtime.h>

#include "jetdl/math/kernel.h"

constexpr int NUM_THREADS_PER_BLOCK = 256;

__global__ void c_mul_cuda_kernel(const float* a, const float* b, float* c,
                                  const size_t N) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    c[i] = a[i] * b[i];
  }
}

void c_mul_cuda(const float* a, const float* b, float* c, const size_t N) {
  const int NUM_BLOCKS_PER_GRID =
      (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  float *device_a, *device_b, *device_c;

  cudaMalloc(&device_a, N * sizeof(float));
  cudaMalloc(&device_b, N * sizeof(float));
  cudaMalloc(&device_c, N * sizeof(float));

  cudaMemcpy(device_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

  c_mul_cuda_kernel<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK>>>(
      device_a, device_b, device_c, N);

  cudaMemcpy(c, device_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
}