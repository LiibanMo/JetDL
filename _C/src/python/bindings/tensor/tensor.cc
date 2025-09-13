#include "jetdl/tensor.h"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "jetdl/python/tensor/bindings.h"
#include "jetdl/python/tensor/methods.h"

namespace py = pybind11;

namespace {

void bind_tensor_class_init(py::class_<jetdl::Tensor> py_tensor) {
  py_tensor.def(py::init<py::object, bool>());
}

void bind_tensor_class_metadata(py::class_<jetdl::Tensor> py_tensor) {
  py_tensor.def_property_readonly(
      "_data", [](jetdl::Tensor& self) { return (*self._data); });
  py_tensor.def_property_readonly("shape", [](const jetdl::Tensor& self) {
    return py::tuple(py::cast(self.shape));
  });
  py_tensor.def_readonly("ndim", &jetdl::Tensor::ndim);
  py_tensor.def_readonly("size", &jetdl::Tensor::size);
  py_tensor.def_property_readonly("strides", [](jetdl::Tensor& self) {
    return py::tuple(py::cast(self.strides));
  });
  py_tensor.def_readonly("is_contiguous", &jetdl::Tensor::is_contiguous);
  py_tensor.def_readonly("requires_grad", &jetdl::Tensor::requires_grad);
  py_tensor.def_property_readonly("grad", [](jetdl::Tensor& self) {
    if (self.grad) {
      return *self.grad;
    } else {
      throw std::runtime_error(".grad attribute null\n");
    }
  });
}

}  // namespace

void bind_tensor_submodule(py::module_& m) {
  py::class_<jetdl::Tensor> py_tensor(m, "Tensor");

  bind_tensor_class_init(py_tensor);
  bind_tensor_class_metadata(py_tensor);

  bind_tensor_math_methods(py_tensor);
  bind_tensor_linalg_methods(py_tensor);
}
