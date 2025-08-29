#pragma once

#include "tensor/tensor.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

struct TensorDeleter {
    void operator()(Tensor* t) const {
        destroy_tensor(t);
    }
};

Tensor* init_tensor(const py::object& data);
void bind_tensor_submodule(py::module_& m);