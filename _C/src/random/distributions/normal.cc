#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "jetdl/random/distributions.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/metadata.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

std::shared_ptr<Tensor> _random_normal(const float mean, const float std,
                                       const std::vector<size_t>& shape,
                                       const size_t seed,
                                       const Device& device) {
  const size_t size = utils::get_size(shape);
  const bool on_cuda = device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    if (!cuda_is_available()) {
      throw std::runtime_error(
          "CUDA is not available. Cannot create tensor on device '" +
          device.str() + "'.");
    }
    float* result_cuda = cuda::CUDAAllocator::allocate(size);
    c_random_normal_cuda(result_cuda, mean, std, size, seed);
    result_tensor = std::make_shared<Tensor>(result_cuda, shape, false, device);
#else
    throw std::runtime_error(
        "JetDL was compiled without CUDA support. "
        "Cannot create tensor on device '" + device.str() + "'.");
#endif
  } else {
    auto generator = std::mt19937_64(seed);
    auto gaussian_dist = std::normal_distribution<float>(mean, std);

    auto result_data = std::shared_ptr<float[]>(new float[size]());
    for (size_t i = 0; i < size; i++) {
      result_data[i] = gaussian_dist(generator);
    }

    result_tensor = std::make_shared<Tensor>(result_data, shape);
  }

  return result_tensor;
}

}  // namespace jetdl
