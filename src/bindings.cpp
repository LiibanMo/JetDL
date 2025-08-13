#include "tensor/bindings.hpp"
#include "linalg/bindings.hpp"
#include "math/bindings.hpp"

PYBIND11_MODULE(_Cpp, m) {
    bind_Tensor_class(m);
       
    linalg::bind_submodule(m);

    math::ops::bind_submodule(m);
    math::function::bind_submodule(m);
    
}