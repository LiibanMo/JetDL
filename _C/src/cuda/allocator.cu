#include "jetdl/cuda/allocator.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <atomic>

namespace jetdl {
namespace cuda {

static std::atomic<size_t> total_allocated_bytes{0};

float* CUDAAllocator::allocate(size_t num_elements) {
  if (num_elements == 0) {
    return nullptr;
  }

  float* ptr = nullptr;
  size_t bytes = num_elements * sizeof(float);

  cudaError_t error = cudaMalloc(&ptr, bytes);
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "CUDA memory allocation failed: " +
        std::string(cudaGetErrorString(error)));
  }

  total_allocated_bytes.fetch_add(bytes);
  return ptr;
}

void CUDAAllocator::deallocate(float* ptr) {
  if (ptr == nullptr) {
    return;
  }

  cudaError_t error = cudaFree(ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "CUDA memory deallocation failed: " +
        std::string(cudaGetErrorString(error)));
  }
}

size_t CUDAAllocator::get_allocated_bytes() {
  return total_allocated_bytes.load();
}

}  // namespace cuda
}  // namespace jetdl
