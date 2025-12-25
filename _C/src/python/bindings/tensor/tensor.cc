#include "jetdl/tensor.h"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "jetdl/device.h"
#include "jetdl/python/tensor/bindings.h"
#include "jetdl/python/tensor/methods.h"

namespace py = pybind11;

namespace jetdl {

namespace {

void bind_tensor_class_init(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def(py::init<Tensor, bool>(), py::arg("tensor"),
                py::arg("requires_grad"));

  py_tensor.def(py::init([](const py::object& data, bool requires_grad,
                            const std::string& device_str) {
                  Device device = Device::parse(device_str);
                  return std::make_shared<Tensor>(data, requires_grad, device);
                }),
                py::arg("data"), py::arg("requires_grad") = false,
                py::arg("device") = "cpu");
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

  // Device property - returns string representation (e.g., "cpu" or "cuda:0")
  py_tensor.def_property_readonly(
      "device",
      [](const std::shared_ptr<Tensor>& self) { return self->device.str(); });

  // Device check properties
  py_tensor.def_property_readonly(
      "is_cuda",
      [](const std::shared_ptr<Tensor>& self) { return self->is_cuda(); });
  py_tensor.def_property_readonly(
      "is_cpu",
      [](const std::shared_ptr<Tensor>& self) { return self->is_cpu(); });
}

void bind_tensor_class_device_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  // .to(device) - transfer tensor to specified device
  py_tensor.def(
      "to",
      [](const std::shared_ptr<Tensor>& self, const std::string& device_str) {
        Device target = Device::parse(device_str);
        return self->to(target);
      },
      py::arg("device"), "Transfer tensor to the specified device");

  // .cuda() - transfer tensor to CUDA device
  py_tensor.def(
      "cuda",
      [](const std::shared_ptr<Tensor>& self, int device_id) {
        return self->cuda(device_id);
      },
      py::arg("device_id") = 0, "Transfer tensor to CUDA device");

  // .cpu() - transfer tensor to CPU
  py_tensor.def(
      "cpu", [](const std::shared_ptr<Tensor>& self) { return self->cpu(); },
      "Transfer tensor to CPU");
}

}  // namespace

void bind_tensor_submodule(py::module_& m) {
  py::class_<Tensor, std::shared_ptr<Tensor>> py_tensor(m, "Tensor",
                                                        py::buffer_protocol());

  bind_tensor_class_init(py_tensor);
  bind_tensor_class_buffer(py_tensor);
  bind_tensor_class_metadata(py_tensor);
  bind_tensor_class_device_methods(py_tensor);

  bind_tensor_autograd_methods(py_tensor);
  bind_tensor_math_methods(py_tensor);
  bind_tensor_linalg_methods(py_tensor);
  bind_tensor_routines_methods(py_tensor);
}

}  // namespace jetdl
