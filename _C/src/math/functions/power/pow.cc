#include <memory>
#include <vector>

#include "jetdl/math/function.h"
#include "jetdl/math/kernel.h"
#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _power(std::shared_ptr<Tensor>& input,
                               const int exponent) {
  std::shared_ptr<std::vector<float>> result_data = input->_data;

  for (size_t i = 0; i < result_data->size(); i++) {
    c_pow_cpu(&result_data->at(i), input->_data->at(i), exponent);
  }

  auto result_tensor =
      std::make_shared<Tensor>(result_data, input->shape, input->requires_grad);

  return result_tensor;
}

}  // namespace jetdl
