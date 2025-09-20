#ifndef JETDL_LINALG_TRANSPOSE_HPP
#define JETDL_LINALG_TRANSPOSE_HPP

#include "jetdl/tensor.h"

namespace jetdl {

std::shared_ptr<Tensor> _linalg_T(std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> _linalg_mT(std::shared_ptr<Tensor>& a);

}  // namespace jetdl

#endif
