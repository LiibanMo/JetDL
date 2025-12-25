#include <memory>
#include <stdexcept>
#include <vector>

#include "jetdl/autograd/routines.h"
#include "jetdl/routines/manipulation.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {
std::shared_ptr<Tensor> _view(std::shared_ptr<Tensor>& input,
                              const std::vector<size_t>& shape) {
  const size_t size = utils::get_size(shape);
  if (size != input->size) {
    throw std::runtime_error("shape incompatible for creating view_tensor->\n");
  }

  // Create tensor that shares data with input, preserving device
  auto result_tensor = std::make_shared<Tensor>(input->_data, shape,
                                                input->requires_grad,
                                                input->device);

  // For CUDA tensors, also share the CUDA data pointer
  if (input->device.is_cuda()) {
    result_tensor->_cuda_data = input->_cuda_data;
  }

  result_tensor->shape = shape;
  result_tensor->strides = utils::get_strides(shape);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<ViewBackward>(input, result_tensor);
  }

  return result_tensor;
}

}  // namespace jetdl