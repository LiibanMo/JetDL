#ifndef JETDL_PYTHON_MATH_BINDINGS_H
#define JETDL_PYTHON_MATH_BINDINGS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jetdl {

void bind_math_submodule(py::module_& m);

}

#endif
