#include "tensor/python/bindings.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_C, m) {
    bind_tensor_submodule(m);
}