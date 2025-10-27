#include "jetdl/optim/kernel.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace jetdl {

void sgd_kernel(float* param, const float lr, const float* param_grad,
                const size_t N) {
  cblas_saxpy(N, -lr, param_grad, 1, param, 1);
}

}  // namespace jetdl
