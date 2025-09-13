#ifndef JETDL_LINALG_PRODUCT_HPP
#define JETDL_LINALG_PRODUCT_HPP

#include "jetdl/tensor.h"

jetdl::Tensor _linalg_dot(const jetdl::Tensor& a, const jetdl::Tensor& b);
jetdl::Tensor _linalg_matvec(const jetdl::Tensor& a, const jetdl::Tensor& b);
jetdl::Tensor _linalg_vecmat(const jetdl::Tensor& a, const jetdl::Tensor& b);
jetdl::Tensor _linalg_matmul(const jetdl::Tensor& a, const jetdl::Tensor& b);

#endif
