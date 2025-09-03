#include "linalg/python/bindings.h"
#include "linalg/linalg.h"

void bind_linalg_submodule(py::module_ &m) {
  py::module_ linalg = m.def_submodule("linalg");

  linalg.def("c_dot", &linalg_dot);
  linalg.def("c_matmul", &linalg_matmul);

  linalg.def("c_T", &linalg_T);
  linalg.def("c_mT", &linalg_mT);
}
