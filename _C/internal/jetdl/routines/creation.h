#ifndef JETDL_ROUTINES_CREATION_HPP
#define JETDL_ROUTINES_CREATION_HPP

#include <memory>
#include <vector>

#include "jetdl/device.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _copy(std::shared_ptr<Tensor>& input);

std::shared_ptr<Tensor> _zeros(const std::vector<size_t>& shape,
                               const bool requires_grad);

std::shared_ptr<Tensor> _ones(const std::vector<size_t>& shape,
                              const bool requires_grad);

std::shared_ptr<Tensor> _fill(const std::vector<size_t>& shape,
                              const float value, const bool requires_grad);

// CUDA kernels for creation functions
void c_fill_cuda(float* d_dest, const float value, const size_t N);
void c_zeros_cuda(float* d_dest, const size_t N);

}  // namespace jetdl

#endif
