#include <cmath>
#include <memory>

#include "jetdl/autograd/math.h"
#include "jetdl/math/function.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _log(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::log(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<LogBackward>(input, result_tensor, 1.0);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _log10(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::log10(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    // ln(10) = 2.302585092994046
    result_tensor->grad_fn =
        std::make_shared<LogBackward>(input, result_tensor, 2.302585092994046);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _log2(std::shared_ptr<Tensor>& input) {
  auto result_data = std::shared_ptr<float[]>(new float[input->size]());

  for (size_t i = 0; i < input->size; i++) {
    result_data[i] = std::log2(input->_data[i]);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  if (result_tensor->requires_grad) {
    // ln(2) = 0.6931471805599453
    result_tensor->grad_fn =
        std::make_shared<LogBackward>(input, result_tensor, 0.6931471805599453);
  }

  return result_tensor;
}

}  // namespace jetdl
