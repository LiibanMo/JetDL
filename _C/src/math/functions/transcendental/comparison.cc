#include <cmath>
#include <memory>

#include "jetdl/autograd/math.h"
#include "jetdl/math/function.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _abs(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::abs(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<AbsBackward>(input, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _sign(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    const float x = input->_data[i];
    result_data[i] = (x > 0.0f) ? 1.0f : (x < 0.0f) ? -1.0f : 0.0f;
  }

  // sign has zero gradient almost everywhere — no backward node needed
  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, false);

  return result_tensor;
}

std::shared_ptr<Tensor> _clamp(std::shared_ptr<Tensor>& input, float min_val,
                                float max_val) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    const float x = input->_data[i];
    result_data[i] = (x < min_val) ? min_val : (x > max_val) ? max_val : x;
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<ClampBackward>(input, result_tensor, min_val, max_val);
  }

  return result_tensor;
}

}  // namespace jetdl
