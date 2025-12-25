#include <cuda_runtime.h>

#include "jetdl/routines/creation.h"

constexpr int NUM_THREADS_PER_BLOCK = 256;

__global__ void c_fill_cuda_kernel(float* dest, const float value,
                                   const size_t N) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    dest[i] = value;
  }
}

void c_fill_cuda(float* d_dest, const float value, const size_t N) {
  const int num_blocks =
      (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  c_fill_cuda_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(d_dest, value, N);
}

void c_zeros_cuda(float* d_dest, const size_t N) {
  cudaMemset(d_dest, 0, N * sizeof(float));
}
