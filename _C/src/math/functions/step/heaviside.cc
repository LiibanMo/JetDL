#include <memory>

#include "jetdl/math/function.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _heaviside_function(std::shared_ptr<Tensor>& input,
                                            const float value) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    const float current_element = input->_data[i];
    if (current_element > 0) {
      result_data[i] = 1.0f;
    } else if (current_element == 0) {
      result_data[i] = value;
    } else if (current_element < 0) {
      result_data[i] = 0.0f;
    }
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        nullptr;  // Need to implement: Dirac Delta function
  }

  return result_tensor;
}

}  // namespace jetdl