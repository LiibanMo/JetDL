#ifndef JETDL_RANDOM_DISTRIBUTIONS_H
#define JETDL_RANDOM_DISTRIBUTIONS_H

#include <memory>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _random_uniform(const float low, const float high,
                                        const std::vector<size_t>& shape,
                                        const size_t seed);

std::shared_ptr<Tensor> _random_normal(const float mean, const float std,
                                       const std::vector<size_t>& shape,
                                       const size_t seed);

}  // namespace jetdl

#endif
