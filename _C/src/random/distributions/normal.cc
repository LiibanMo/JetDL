#include <memory>
#include <random>
#include <vector>

#include "jetdl/random/distributions.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> _random_normal(const float mean, const float std,
                                       const std::vector<size_t>& shape,
                                       const size_t seed) {
  auto generator = std::mt19937_64(seed);

  auto gaussian_dist = std::normal_distribution<float>(mean, std);

  const size_t result_size = utils::get_size(shape);
  auto result_data = std::shared_ptr<float[]>(new float[result_size]());

  for (size_t i = 0; i < result_size; i++) {
    result_data[i] = gaussian_dist(generator);
  }

  auto result_tensor = std::make_shared<Tensor>(result_data, shape);

  return result_tensor;
}

}  // namespace jetdl
