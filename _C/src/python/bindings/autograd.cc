#include <pybind11/pybind11.h>

#include "jetdl/autograd/graph.h"
#include "jetdl/tensor.h"

namespace py = pybind11;

void bind_autograd_submodule(py::module_& m) {
  py::module_ autograd = m.def_submodule("autograd");

  autograd.def("c_backward", [](jetdl::Tensor& tensor) {
    jetdl::autograd::Graph(tensor).backward();
  });
}
