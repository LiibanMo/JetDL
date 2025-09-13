#include "jetdl/linalg.h"

#include <pybind11/pybind11.h>

#include "jetdl/python/linalg/bindings.h"

namespace py = pybind11;

void bind_linalg_submodule(py::module_& m) {
  py::module_ linalg = m.def_submodule("linalg");

  linalg.def("c_dot", &jetdl::linalg::dot);
  linalg.def("c_matmul", &jetdl::linalg::matmul);

  linalg.def("c_T", &jetdl::linalg::T);
  linalg.def("c_mT", &jetdl::linalg::mT);
}
