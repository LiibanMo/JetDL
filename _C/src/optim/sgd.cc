#include <memory>
#include <stdexcept>

#include "jetdl/optim/kernel.h"
#include "jetdl/optim/optim.h"
#include "jetdl/routines.h"

namespace jetdl {

namespace optim {

void sgd_step(std::shared_ptr<Tensor>& param, const float lr) {
  if (!param->grad) {
    throw std::runtime_error(
        "INTERNAL (sgd.cc::sgd_step): GRAD DOES NOT EXIST\n");
  }
  if (param->size != param->grad->size) {
    throw std::runtime_error(
        "INTERNAL (sgd.cc::sgd_step) param->size != param->grad->size\n");
  }

  std::shared_ptr<Tensor> param_grad = contiguous(param->grad);

  sgd_kernel(param->get(), lr, param_grad->get(), param->size);
}

}  // namespace optim

}  // namespace jetdl
