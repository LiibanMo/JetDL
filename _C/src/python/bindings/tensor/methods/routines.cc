#include "jetdl/routines.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace jetdl {

namespace {

void bind_tensor_method_reshape(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("reshape", &jetdl::reshape);
}

void bind_tensor_method_squeeze(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def(
      "squeeze", [](std::shared_ptr<Tensor>& self, const py::object& axes) {
        if (axes.is_none()) {
          return jetdl::squeeze(self, {});
        } else if (py::isinstance<py::list>(axes) ||
                   py::isinstance<py::tuple>(axes)) {
          return jetdl::squeeze(self, py::cast<std::vector<int>>(axes));
        } else if (py::isinstance<py::int_>(axes)) {
          const auto& input_axis = std::vector<int>{py::cast<int>(axes)};
          return jetdl::squeeze(self, input_axis);
        } else {
          throw py::type_error(
              py::str("type '{}' not valid as axes input when squeezing tensor")
                  .format(py::type::of(axes)));
        }
      });
}

void bind_tensor_method_unsqueeze(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("unsqueeze", [](std::shared_ptr<Tensor>& self,
                                const py::object& axis) {
    if (!py::isinstance<py::int_>(axis)) {
      throw py::type_error(
          py::str("type '{}' not valid as axes input when unsqueezing tensor")
              .format(py::type::of(axis)));
    }

    const int input_axis = py::cast<int>(axis);
    return jetdl::unsqueeze(self, input_axis);
  });
}

}  // namespace

void bind_tensor_routines_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  bind_tensor_method_unsqueeze(py_tensor);
  bind_tensor_method_squeeze(py_tensor);
  bind_tensor_method_reshape(py_tensor);
}

}  // namespace jetdl
