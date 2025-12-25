#ifndef JETDL_CUDA_ALLOCATOR_H
#define JETDL_CUDA_ALLOCATOR_H

#include <cstddef>

namespace jetdl {
namespace cuda {

class CUDAAllocator {
 public:
  static float* allocate(size_t num_elements);

  static void deallocate(float* ptr);

  static size_t get_allocated_bytes();
};

}  // namespace cuda
}  // namespace jetdl

#endif
