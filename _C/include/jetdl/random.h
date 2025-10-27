#ifndef JETDL_RANDOM_H
#define JETDL_RANDOM_H

#include <cstddef>
#include <memory>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {
namespace random {

std::shared_ptr<Tensor> normal(const float mean, const float std,
                               const std::vector<size_t>& shape,
                               const size_t seed = 123);

std::shared_ptr<Tensor> uniform(const float low, const float high,
                                const std::vector<size_t>& shape,
                                const size_t seed = 123);

}  // namespace random
}  // namespace jetdl
#endif
