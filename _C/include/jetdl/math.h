#ifndef JETDL_MATH_H
#define JETDL_MATH_H

#include <memory>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {
namespace math {

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);

void add_inplace(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b);

std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);

std::shared_ptr<Tensor> div(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);

std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor>& a, const double exponent);

std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor>& a,
                            const std::vector<int>& axes = {});

std::shared_ptr<Tensor> sum_to_shape(std::shared_ptr<Tensor>& tensor,
                                     const std::vector<size_t>& shape);

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor>& a,
                             const std::vector<int>& axes = {});

std::shared_ptr<Tensor> heaviside(std::shared_ptr<Tensor>& a,
                                  const float value = 0.0f);

}  // namespace math
}  // namespace jetdl

#endif
