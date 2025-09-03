#include "routines/python/bindings.h"
#include "routines/routines.h"

#include <pybind11/stl.h>

void bind_routines_submodule(py::module_ &m) {
  py::module_ routines = m.def_submodule("routines");
  routines.def("c_ones", &routines_ones);
}
