#include <memory>

#include "jetdl/math.h"
#include "jetdl/nn.h"

namespace jetdl {

namespace nn {

std::shared_ptr<Tensor> MSELoss::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs) {
  std::shared_ptr<Tensor>& y0 = inputs[0];
  std::shared_ptr<Tensor>& y1 = inputs[1];

  std::shared_ptr<Tensor> diff = math::sub(y0, y1);
  std::shared_ptr<Tensor> diff_squared = math::pow(diff, 2);
  std::shared_ptr<Tensor> mse = math::mean(diff_squared);
  return mse;
}

}  // namespace nn

}  // namespace jetdl
