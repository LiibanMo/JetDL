#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace linalg {
    void bind_dot(py::module_& m);
    void bind_matmul(py::module_& m);
    void bind_T(py::module_& m);
    void bind_mT(py::module_& m);
}