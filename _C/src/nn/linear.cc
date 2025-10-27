#include <memory>

#include "jetdl/linalg.h"
#include "jetdl/math.h"
#include "jetdl/nn.h"
#include "jetdl/tensor.h"

namespace jetdl {

namespace nn {

std::shared_ptr<Tensor> linear_forward(std::shared_ptr<Tensor>& input,
                                       std::shared_ptr<Tensor>& weight,
                                       std::shared_ptr<Tensor>& bias) {
  std::shared_ptr<Tensor> weight_mT = linalg::mT(weight);
  std::shared_ptr<Tensor> result1 = linalg::matmul(input, weight_mT);
  std::shared_ptr<Tensor> result = math::add(result1, bias);
  return result;
}

}  // namespace nn

}  // namespace jetdl