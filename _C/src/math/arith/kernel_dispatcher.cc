#include "jetdl/math/kernel.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/utils/device.h"
#endif

void c_add_kernel(const float* a, const float* b, float* c, const size_t N) {
#ifdef JETDL_WITH_CUDA
  if (is_cuda_available) {
    c_add_cuda(a, b, c, N);
  } else {
    c_add_cpu(a, b, c, N);
  }
#else
  c_add_cpu(a, b, c, N);
#endif
}