#ifndef JETDL_NN_H
#define JETDL_NN_H

#include <memory>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {

namespace nn {

enum class WeightInitType { HE, LECUN, XAVIER, ZERO, RANDOM };

class Parameter : public Tensor {
 public:
  Parameter(const std::vector<size_t>& shape,
            const WeightInitType& weight_init_type);
};

class Module {
 public:
  ~Module() = default;
  virtual std::shared_ptr<Tensor> forward(
      std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
};

class Linear : public Module {
 private:
  std::shared_ptr<Tensor> weights = nullptr;
  std::shared_ptr<Tensor> bias = nullptr;

 public:
  Linear(const size_t in_features, const size_t out_features,
         const WeightInitType& weight_init_type = WeightInitType::HE);

  std::shared_ptr<Tensor> forward(
      std::vector<std::shared_ptr<Tensor>>& inputs) override;
};

class MSELoss : public Module {
 public:
  std::shared_ptr<Tensor> forward(
      std::vector<std::shared_ptr<Tensor>>& inputs) override;
};

}  // namespace nn

}  // namespace jetdl

#endif
