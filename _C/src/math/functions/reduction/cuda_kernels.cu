#include <cuda_runtime.h>

#include "jetdl/math/kernel.h"

constexpr int NUM_THREADS_PER_BLOCK = 256;

// ============================================================================
// Total Sum Reduction
// ============================================================================

__global__ void c_total_sum_cuda_kernel(float* dest, const float* src,
                                        const size_t N) {
  __shared__ float shared_data[NUM_THREADS_PER_BLOCK];

  const int tid = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  // Each thread accumulates its portion of the array
  float thread_sum = 0.0f;
  for (size_t i = global_id; i < N; i += grid_size) {
    thread_sum += src[i];
  }

  shared_data[tid] = thread_sum;
  __syncthreads();

  // Tree-based reduction within the block
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    __syncthreads();
  }

  // First thread in each block adds to global result
  if (tid == 0) {
    atomicAdd(dest, shared_data[0]);
  }
}

void c_total_sum_cuda(float* d_dest, const float* d_src, const size_t N) {
  // Initialize destination to zero
  cudaMemset(d_dest, 0, sizeof(float));

  // Calculate grid size - limit blocks to avoid over-subscription
  const int num_blocks = min((int)((N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK), 256);

  c_total_sum_cuda_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(d_dest, d_src, N);
}

// ============================================================================
// Sum Over Axes (scatter-add pattern)
// ============================================================================

__global__ void c_sum_over_axes_cuda_kernel(float* dest, const float* src,
                                            const size_t* dest_idxs,
                                            const size_t N) {
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id < N) {
    atomicAdd(&dest[dest_idxs[global_id]], src[global_id]);
  }
}

void c_sum_over_axes_cuda(float* d_dest, const float* d_src,
                          const size_t* h_dest_idxs, const size_t result_size,
                          const size_t N) {
  // Initialize destination to zero
  cudaMemset(d_dest, 0, result_size * sizeof(float));

  // Copy dest_idxs from host to device
  size_t* d_dest_idxs;
  cudaMalloc(&d_dest_idxs, N * sizeof(size_t));
  cudaMemcpy(d_dest_idxs, h_dest_idxs, N * sizeof(size_t),
             cudaMemcpyHostToDevice);

  const int num_blocks =
      (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  c_sum_over_axes_cuda_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(
      d_dest, d_src, d_dest_idxs, N);

  cudaFree(d_dest_idxs);
}

// ============================================================================
// Total Mean Reduction
// ============================================================================

__global__ void c_total_mean_cuda_kernel(float* dest, const float* src,
                                         const size_t N) {
  __shared__ float shared_data[NUM_THREADS_PER_BLOCK];

  const int tid = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  // Each thread accumulates its portion of the array
  float thread_sum = 0.0f;
  for (size_t i = global_id; i < N; i += grid_size) {
    thread_sum += src[i];
  }

  shared_data[tid] = thread_sum;
  __syncthreads();

  // Tree-based reduction within the block
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    __syncthreads();
  }

  // First thread in each block adds to global result
  if (tid == 0) {
    atomicAdd(dest, shared_data[0]);
  }
}

__global__ void c_divide_by_n_kernel(float* dest, const size_t N) {
  *dest /= N;
}

void c_total_mean_cuda(float* d_dest, const float* d_src, const size_t N) {
  // Initialize destination to zero
  cudaMemset(d_dest, 0, sizeof(float));

  // Calculate grid size
  const int num_blocks = min((int)((N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK), 256);

  // Sum all elements
  c_total_mean_cuda_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(d_dest, d_src, N);

  // Divide by N to get mean
  c_divide_by_n_kernel<<<1, 1>>>(d_dest, N);
}

// ============================================================================
// Mean Over Axes (scatter-add with division)
// ============================================================================

__global__ void c_mean_over_axes_cuda_kernel(float* dest, const float* src,
                                             const size_t* dest_idxs,
                                             const float divisor,
                                             const size_t N) {
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id < N) {
    atomicAdd(&dest[dest_idxs[global_id]], src[global_id] / divisor);
  }
}

void c_mean_over_axes_cuda(float* d_dest, const float* d_src,
                           const size_t* h_dest_idxs, const size_t result_size,
                           const size_t divisor, const size_t N) {
  // Initialize destination to zero
  cudaMemset(d_dest, 0, result_size * sizeof(float));

  // Copy dest_idxs from host to device
  size_t* d_dest_idxs;
  cudaMalloc(&d_dest_idxs, N * sizeof(size_t));
  cudaMemcpy(d_dest_idxs, h_dest_idxs, N * sizeof(size_t),
             cudaMemcpyHostToDevice);

  const int num_blocks =
      (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  c_mean_over_axes_cuda_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(
      d_dest, d_src, d_dest_idxs, static_cast<float>(divisor), N);

  cudaFree(d_dest_idxs);
}
