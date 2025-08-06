#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace math {

    void bind_add(py::module_& m);
    void bind_sub(py::module_& m);
    void bind_mul(py::module_& m);
    void bind_div(py::module_& m);

}