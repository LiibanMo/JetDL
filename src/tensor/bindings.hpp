#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_Tensor_class(py::module_& m);