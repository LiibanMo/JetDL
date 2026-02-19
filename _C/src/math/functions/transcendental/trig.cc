#include <cmath>
#include <memory>

#include "jetdl/autograd/math.h"
#include "jetdl/math/function.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _sin(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::sin(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<SinBackward>(input, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _cos(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::cos(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<CosBackward>(input, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _tanh(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::tanh(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<TanhBackward>(input, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _sinh(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::sinh(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<SinhBackward>(input, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _cosh(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::cosh(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<CoshBackward>(input, result_tensor);
  }

  return result_tensor;
}

}  // namespace jetdl
