#include "product.hpp"
#include "product/matmul.hpp"

namespace linalg {

    Tensor dot(const Tensor& a, const Tensor& b) {
        return c_dot(a, b);
    }
    
    Tensor matmul(const Tensor& a, const Tensor& b) {
        Tensor result_tensor;
        if (a.ndim == 1 && b.ndim == 1) {
            result_tensor = c_dot(a, b);
        } else if (a.ndim > 1 && b.ndim == 1) {
            result_tensor = c_matvec(a, b);
        } else if (a.ndim == 1 && b.ndim > 1) {
            result_tensor = c_vecmat(a, b);
        } else if (a.ndim > 1 && b.ndim > 1) {
            result_tensor = c_matmul(a, b);
        }
        return result_tensor;
    }

}