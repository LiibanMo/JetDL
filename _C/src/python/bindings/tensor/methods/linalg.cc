#include "jetdl/linalg.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "jetdl/python/tensor/methods.h"
#include "jetdl/tensor.h"

namespace py = pybind11;

namespace jetdl {

namespace {

void bind_tensor_matmul_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__matmul__", [](std::shared_ptr<Tensor>& self,
                                 std::shared_ptr<Tensor>& other) {
    return linalg::matmul(self, other);
  });
}

void bind_tensor_T_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def_property_readonly(
      "T", [](std::shared_ptr<Tensor>& self) { return linalg::T(self); });
}

void bind_tensor_mT_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def_property_readonly(
      "mT", [](std::shared_ptr<Tensor>& self) { return linalg::mT(self); });
}

}  // namespace

void bind_tensor_linalg_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  bind_tensor_matmul_method(py_tensor);
  bind_tensor_T_method(py_tensor);
  bind_tensor_mT_method(py_tensor);
}

}  // namespace jetdl
