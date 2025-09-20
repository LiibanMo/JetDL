#ifndef JETDL_LINALG_PRODUCT_HPP
#define JETDL_LINALG_PRODUCT_HPP

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _linalg_dot(std::shared_ptr<Tensor>& a,
                                    std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> _linalg_matvec(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> _linalg_vecmat(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> _linalg_matmul(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b);

}  // namespace jetdl

#endif
