#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

#include "jetdl/autograd/graph.h"
#include "jetdl/python/tensor/methods.h"

namespace py = pybind11;

namespace jetdl {

namespace {

void bind_tensor_backward_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("backward", [](std::shared_ptr<Tensor>& self) {
    if (self->ndim != 0) {
      throw std::runtime_error(
          py::str(
              "backward pass only starts for scalar tensors; got {}-D tensor\n")
              .format(self->ndim));
    }
    self->grad = std::make_shared<Tensor>(1.0f, false);

    auto graph = Graph();
    graph.traverse(self);
    graph.apply();
  });
}

}  // namespace

void bind_tensor_autograd_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  bind_tensor_backward_method(py_tensor);
}

}  // namespace jetdl
