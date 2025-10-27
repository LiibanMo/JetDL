#ifndef JETDL_OPTIM_KERNEL_H
#define JETDL_OPTIM_KERNEL_H

#include <cstddef>

namespace jetdl {

void sgd_kernel(float* param, const float lr, const float* param_grad,
                const size_t N);

}

#endif