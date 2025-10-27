#include "jetdl/nn.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jetdl {

namespace {

void bind_linear_forward(py::module_& m) {
  m.def("c_linear_forward", &nn::linear_forward);
}

void bind_relu_forward(py::module_& m) {
  m.def("c_relu_forward", &nn::relu_forward);
}

}  // namespace

void bind_nn_submodule(py::module_& m) {
  py::module_ nn = m.def_submodule("nn");
  bind_linear_forward(nn);
  bind_relu_forward(nn);
}

}  // namespace jetdl