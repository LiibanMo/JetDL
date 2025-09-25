#include <memory>
#include <vector>

#include "jetdl/linalg.h"
#include "jetdl/math.h"
#include "jetdl/nn.h"
#include "jetdl/tensor.h"

namespace jetdl {

namespace nn {

Linear::Linear(const size_t in_features, const size_t out_features,
               const WeightInitType& weight_init_type) {
  const auto& weights_shape = std::vector<size_t>{in_features, out_features};
  this->weights = std::make_shared<Parameter>(weights_shape, weight_init_type);

  const auto& bias_shape = std::vector<size_t>{out_features};
  this->bias = std::make_shared<Parameter>(bias_shape, weight_init_type);
}

std::shared_ptr<Tensor> Linear::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs) {
  std::shared_ptr<Tensor>& input = inputs[0];
  std::shared_ptr<Tensor> intermediate = linalg::matmul(input, this->weights);
  std::shared_ptr<Tensor> output = math::add(intermediate, this->bias);
  return math::add(output, this->bias);
}

}  // namespace nn

}  // namespace jetdl
