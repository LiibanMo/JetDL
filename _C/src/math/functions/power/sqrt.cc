#include <memory>

#include "jetdl/autograd/math.h"
#include "jetdl/math/function.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _square_root(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::sqrt(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<PowBackward>(input, 1 / 2, result_tensor);
  }

  return result_tensor;
}

}  // namespace jetdl
