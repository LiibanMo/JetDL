#ifndef JETDL_LINALG_H
#define JETDL_LINALG_H

#include "jetdl/tensor.h"

namespace jetdl {
namespace linalg {

jetdl::Tensor dot(const jetdl::Tensor& a, const jetdl::Tensor& b);

jetdl::Tensor matmul(const jetdl::Tensor& a, const jetdl::Tensor& b);

jetdl::Tensor T(const jetdl::Tensor& a);

jetdl::Tensor mT(const jetdl::Tensor& a);

}  // namespace linalg
}  // namespace jetdl
#endif
