#ifndef JETDL_PYTHON_RANDOM_BINDINGS_H
#define JETDL_PYTHON_RANDOM_BINDINGS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jetdl {

void bind_random_submodule(py::module_& m);

}

#endif
