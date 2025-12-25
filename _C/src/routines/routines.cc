#include "jetdl/routines.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "jetdl/routines/creation.h"
#include "jetdl/routines/manipulation.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/check.h"
#include "jetdl/utils/metadata.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

std::shared_ptr<Tensor> copy(std::shared_ptr<Tensor>& input) {
  return _copy(input);
}

std::shared_ptr<Tensor> zeros(const std::vector<size_t>& shape,
                              const bool requires_grad,
                              const Device& device) {
  return fill(shape, 0.0f, requires_grad, device);
}

std::shared_ptr<Tensor> ones(const std::vector<size_t>& shape,
                             const bool requires_grad,
                             const Device& device) {
  return fill(shape, 1.0f, requires_grad, device);
}

std::shared_ptr<Tensor> fill(const std::vector<size_t>& shape,
                             const float value, const bool requires_grad,
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
    if (value == 0.0f) {
      c_zeros_cuda(result_cuda, size);
    } else {
      c_fill_cuda(result_cuda, value, size);
    }
    result_tensor =
        std::make_shared<Tensor>(result_cuda, shape, requires_grad, device);
#else
    throw std::runtime_error(
        "JetDL was compiled without CUDA support. "
        "Cannot create tensor on device '" + device.str() + "'.");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[size]());
    std::fill(result_data.get(), result_data.get() + size, value);
    result_tensor =
        std::make_shared<Tensor>(result_data, shape, requires_grad);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor>& tensor,
                                const std::vector<int>& shape) {
  utils::check_reshape_shape(shape, tensor->size);
  auto new_shape = std::vector<size_t>(shape.size(), 0);
  std::copy(shape.begin(), shape.end(), new_shape.begin());
  return view(tensor, new_shape);
}

std::shared_ptr<Tensor> view(std::shared_ptr<Tensor>& tensor,
                             const std::vector<size_t>& shape,
                             const bool requires_grad) {
  return _view(tensor, shape);
}

std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor>& input,
                                const std::vector<int>& axes) {
  utils::check_axes(input->shape, axes);
  return _squeeze(input, axes);
}

std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor>& input,
                                  const int axes) {
  utils::check_axes(input->shape, std::vector<int>{axes},
                    utils::SubModule::ROUTINES);
  return _unsqueeze(input, axes);
}

std::shared_ptr<Tensor> contiguous(std::shared_ptr<Tensor>& input) {
  return _make_contiguous(input);
}

std::string tensor_to_string(const std::shared_ptr<Tensor>& input) {
  return _tensor_to_string(input);
}

}  // namespace jetdl
