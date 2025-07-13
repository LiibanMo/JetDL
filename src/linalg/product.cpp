#include "product.hpp"
#include "product/matmul.hpp"

Tensor c_matmul(Tensor& a, Tensor& b) {
    Tensor result_tensor;
    if (a.ndim > 2 || b.ndim >= 2) {
        result_tensor = c_matmul_batched(a, b);
    }
    return result_tensor;
}