#include "python/bindings.h"
#include "linalg/python/bindings.h"
#include "tensor/python/bindings.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_C, m) {
    bind_tensor_submodule(m);
    bind_linalg_submodule(m);
    bind_utils_submodule(m);
}