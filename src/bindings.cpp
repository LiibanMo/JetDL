#include "tensor/bindings.hpp"
#include "linalg/bindings.hpp"
#include "math/bindings.hpp"

PYBIND11_MODULE(_Cpp, m) {
    bind_Tensor_class(m);
       
    linalg::bind_dot(m);
    linalg::bind_matmul(m);
    linalg::bind_T(m);
    linalg::bind_mT(m);

    math::bind_add(m);
    math::bind_sub(m);
    math::bind_mul(m);
    math::bind_div(m);
    
}