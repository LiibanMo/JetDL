#include "jetdl/random.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "jetdl/python/random/bindings.h"

namespace py = pybind11;

namespace jetdl {

void bind_random_submodule(py::module_& m) {
  py::module_ random = m.def_submodule("random");
  random.def("c_uniform", &random::uniform);
  random.def("c_normal", &random::normal);
}

}  // namespace jetdl
