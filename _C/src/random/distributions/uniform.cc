#include <memory>
#include <random>
#include <vector>

#include "jetdl/random/distributions.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> _random_uniform(const float low, const float high,
                                        const std::vector<size_t>& shape,
                                        const size_t seed) {
  auto generator = std::mt19937(seed);

  auto uniform_dist = std::uniform_real_distribution<float>(low, high);

  const size_t size = utils::get_size(shape);
  auto result_data = std::shared_ptr<float[]>(new float[size]());

  for (size_t i = 0; i < size; i++) {
    result_data[i] = uniform_dist(generator);
  }

  auto result_tensor = std::make_shared<Tensor>(result_data, shape);

  return result_tensor;
}

}  // namespace jetdl
