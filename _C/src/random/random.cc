#include "jetdl/random.h"

#include <cstddef>
#include <stdexcept>
#include <memory>
#include <vector>

#include "jetdl/random/distributions.h"
#include "jetdl/tensor.h"

namespace jetdl {
namespace random {

std::shared_ptr<Tensor> normal(const float mean, const float std,
                               const std::vector<size_t>& shape,
                               const size_t seed,
                               const Device& device) {
  if (std <= 0) {
    throw py::type_error(
        py::str("std must be greater than 0. Got {}").format(std));
  }
  return _random_normal(mean, std, shape, seed, device);
}

std::shared_ptr<Tensor> uniform(const float low, const float high,
                                const std::vector<size_t>& shape,
                                const size_t seed,
                                const Device& device) {
  if (low >= high) {
    throw std::runtime_error(
        py::str("interval for uniform dist ({}, {}) is invalid")
            .format(low, high));
  }

  return _random_uniform(low, high, shape, seed, device);
}

}  // namespace random
}  // namespace jetdl
