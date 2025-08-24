#include "linalg.h"
#include "product/matmul.h"

Tensor& linalg_dot(const Tensor& a, const Tensor& b) {
    return *c_linalg_dot(&a, &b);
}