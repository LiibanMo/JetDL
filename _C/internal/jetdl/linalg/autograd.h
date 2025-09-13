#ifndef JETDL_LINALG_AUTOGRAD_HPP
#define JETDL_LINALG_AUTOGRAD_HPP

#include "jetdl/tensor.h"

namespace jetdl {
namespace linalg {

void dot_apply(jetdl::Tensor& tensor);
void vecmat_apply(jetdl::Tensor& tensor);
void matvec_apply(jetdl::Tensor& tensor);
void matmul_apply(jetdl::Tensor& tensor);

}  // namespace linalg
}  // namespace jetdl

#endif
