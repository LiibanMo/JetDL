#ifndef JETDL_MATH_H
#define JETDL_MATH_H

#include "jetdl/tensor.h"

namespace jetdl {
namespace math {

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor>& a,
                            const std::vector<int>& axes);
std::shared_ptr<Tensor> sum_to_shape(std::shared_ptr<Tensor>& tensor,
                                     const std::vector<size_t>& shape);

}  // namespace math
}  // namespace jetdl

#endif
