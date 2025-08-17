#include "tensor/bindings.hpp"
#include "autograd/bindings.hpp"
#include "linalg/bindings.hpp"
#include "math/bindings.hpp"
#include "routines/bindings.hpp"

PYBIND11_MODULE(_Cpp, m) {

    bind_tensor_class(m);
    autograd::bind_submodule(m);
    linalg::bind_submodule(m);
    math::ops::bind_submodule(m);
    math::function::bind_submodule(m);
    routines::bind_submodule(m);
    
}