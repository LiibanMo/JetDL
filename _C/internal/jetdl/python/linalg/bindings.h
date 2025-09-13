#ifndef JETDL_PYTHON_LINALG_BINDINGS_H
#define JETDL_PYTHON_LINALG_BINDINGS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_linalg_submodule(py::module_& m);

#endif
