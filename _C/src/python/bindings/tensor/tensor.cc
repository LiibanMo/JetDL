#include "jetdl/tensor.h"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "jetdl/python/tensor/bindings.h"
#include "jetdl/python/tensor/methods.h"

namespace py = pybind11;

namespace jetdl {

namespace {

void bind_tensor_class_init(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def(py::init<Tensor, bool>(), py::arg("tensor"),
                py::arg("requires_grad"));

  py_tensor.def(py::init<py::object, bool>(), py::arg("data"),
                py::arg("requires_grad"));
}

void bind_tensor_class_buffer(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def_buffer([](const Tensor& self) {
    auto byte_strides = std::vector<size_t>(self.ndim);
    for (size_t i = 0; i < self.ndim; i++) {
      byte_strides[i] = self.strides[i] * sizeof(float);
    }
    return py::buffer_info(self._data.get(), sizeof(float),
                           py::format_descriptor<float>::format(), self.ndim,
                           self.shape, byte_strides);
  });
}

void bind_tensor_class_metadata(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def_property_readonly("shape",
                                  [](const std::shared_ptr<Tensor>& self) {
                                    return py::tuple(py::cast(self->shape));
                                  });
  py_tensor.def_readonly("ndim", &Tensor::ndim);
  py_tensor.def_readonly("size", &Tensor::size);
  py_tensor.def_property_readonly("strides", [](Tensor& self) {
    return py::tuple(py::cast(self.strides));
  });
  py_tensor.def_readonly("is_contiguous", &Tensor::is_contiguous);
  py_tensor.def_readonly("requires_grad", &Tensor::requires_grad);
  py_tensor.def_property_readonly(
      "grad", [](const std::shared_ptr<Tensor>& self) { return self->grad; });
}

}  // namespace

void bind_tensor_submodule(py::module_& m) {
  py::class_<Tensor, std::shared_ptr<Tensor>> py_tensor(m, "Tensor",
                                                        py::buffer_protocol());

  bind_tensor_class_init(py_tensor);
  bind_tensor_class_buffer(py_tensor);
  bind_tensor_class_metadata(py_tensor);

  bind_tensor_autograd_methods(py_tensor);
  bind_tensor_math_methods(py_tensor);
  bind_tensor_linalg_methods(py_tensor);
  bind_tensor_routines_methods(py_tensor);
}

}  // namespace jetdl
