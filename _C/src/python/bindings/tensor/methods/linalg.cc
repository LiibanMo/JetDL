#include "jetdl/linalg.h"

#include <pybind11/pybind11.h>

#include "jetdl/python/tensor/methods.h"
#include "jetdl/tensor.h"

namespace py = pybind11;

namespace {
void bind_tensor_matmul_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def("__matmul__",
                [](const jetdl::Tensor& self, const jetdl::Tensor& other) {
                  return jetdl::linalg::matmul(self, other);
                });
}

void bind_tensor_T_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def_property_readonly(
      "T", [](const jetdl::Tensor& self) { return jetdl::linalg::T(self); });
}

void bind_tensor_mT_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def_property_readonly(
      "T", [](const jetdl::Tensor& self) { return jetdl::linalg::mT(self); });
}

}  // namespace

void bind_tensor_linalg_methods(py::class_<jetdl::Tensor>& py_tensor) {
  bind_tensor_matmul_method(py_tensor);
  bind_tensor_T_method(py_tensor);
  bind_tensor_mT_method(py_tensor);
}
