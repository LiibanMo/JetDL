#include "jetdl/routines.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "jetdl/python/routines/bindings.h"

namespace py = pybind11;

void bind_routines_submodule(py::module_& m) {
  py::module_ routines = m.def_submodule("routines");
  routines.def("c_ones", &jetdl::ones);
}
