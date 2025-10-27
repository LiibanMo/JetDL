#ifndef JETDL_PYTHON_OPTIM_BINDINGS_H
#define JETDL_PYTHON_OPTIM_BINDINGS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jetdl {

void bind_optim_submodule(py::module_& m);

}

#endif