#ifndef JETDL_RANDOM_DISTRIBUTIONS_H
#define JETDL_RANDOM_DISTRIBUTIONS_H

#include <memory>
#include <vector>

#include "jetdl/device.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _random_uniform(const float low, const float high,
                                        const std::vector<size_t>& shape,
                                        const size_t seed,
                                        const Device& device);

std::shared_ptr<Tensor> _random_normal(const float mean, const float std,
                                       const std::vector<size_t>& shape,
                                       const size_t seed,
                                       const Device& device);

// CUDA random number generation using cuRAND
void c_random_uniform_cuda(float* d_dest, const float low, const float high,
                           const size_t N, const size_t seed);

void c_random_normal_cuda(float* d_dest, const float mean, const float std,
                          const size_t N, const size_t seed);

}  // namespace jetdl

#endif
