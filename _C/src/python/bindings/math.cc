#include "jetdl/math.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "jetdl/python/math/bindings.h"

namespace py = pybind11;

namespace jetdl {

void bind_math_submodule(py::module_& m) {
  py::module_ math = m.def_submodule("math");
  math.def("c_add", &math::add, py::call_guard<py::gil_scoped_release>());
  math.def("c_sub", &math::sub, py::call_guard<py::gil_scoped_release>());
  math.def("c_mul", &math::mul, py::call_guard<py::gil_scoped_release>());
  math.def("c_div", &math::div, py::call_guard<py::gil_scoped_release>());
  math.def("c_pow", &math::pow);
  math.def("c_scalar_sqrt", [](py::object& input) {
    if (py::isinstance<py::int_>(input) || py::isinstance<py::float_>(input)) {
      return std::sqrt(py::cast<float>(input));
    } else {
      throw std::runtime_error(
          "INTERNAL: c_scalar_sqrt outside int & float not supported.\n");
    }
  });
  math.def("c_sqrt", &math::sqrt);
  math.def("c_sum",
           [](std::shared_ptr<Tensor>& tensor, const py::object& axes) {
             if (py::isinstance<py::int_>(axes)) {
               auto axes_vec = std::vector<int>(1, py::cast<int>(axes));
               return math::sum(tensor, axes_vec);
             } else if (py::isinstance<py::list>(axes) ||
                        py::isinstance<py::tuple>(axes)) {
               auto axes_vec = axes.cast<std::vector<int>>();
               return math::sum(tensor, axes_vec);
             } else if (axes.is_none()) {
               auto axes_vec = std::vector<int>{};
               return math::sum(tensor, axes_vec);
             } else {
               throw std::runtime_error(
                   py::str("Invalid data type for axes. Got type '{}'")
                       .format(py::type::of(axes)));
             }
           });

  math.def("c_sum_to_shape", [](std::shared_ptr<Tensor>& self,
                                const py::object& shape) {
    if (py::isinstance<py::list>(shape) || py::isinstance<py::tuple>(shape)) {
      auto shape_vec = py::cast<std::vector<size_t>>(shape);
      return math::sum_to_shape(self, shape_vec);
    } else {
      throw py::type_error(
          py::str("type '{}' for shape input for tensor reduction.")
              .format(py::type::of(shape)));
    }
  });

  math.def("c_exp", &math::exp, py::call_guard<py::gil_scoped_release>());
  math.def("c_log", &math::log, py::call_guard<py::gil_scoped_release>());
  math.def("c_log10", &math::log10, py::call_guard<py::gil_scoped_release>());
  math.def("c_log2", &math::log2, py::call_guard<py::gil_scoped_release>());
  math.def("c_sin", &math::sin, py::call_guard<py::gil_scoped_release>());
  math.def("c_cos", &math::cos, py::call_guard<py::gil_scoped_release>());
  math.def("c_tanh", &math::tanh, py::call_guard<py::gil_scoped_release>());
  math.def("c_sinh", &math::sinh, py::call_guard<py::gil_scoped_release>());
  math.def("c_cosh", &math::cosh, py::call_guard<py::gil_scoped_release>());
  math.def("c_abs", &math::abs, py::call_guard<py::gil_scoped_release>());
  math.def("c_sign", &math::sign, py::call_guard<py::gil_scoped_release>());
  math.def("c_clamp", &math::clamp, py::call_guard<py::gil_scoped_release>());

  math.def("c_mean",
           [](std::shared_ptr<Tensor>& tensor, const py::object& axes) {
             if (py::isinstance<py::int_>(axes)) {
               auto axes_vec = std::vector<int>(1, py::cast<int>(axes));
               return math::mean(tensor, axes_vec);
             } else if (py::isinstance<py::list>(axes) ||
                        py::isinstance<py::tuple>(axes)) {
               auto axes_vec = axes.cast<std::vector<int>>();
               return math::mean(tensor, axes_vec);
             } else if (axes.is_none()) {
               auto axes_vec = std::vector<int>{};
               return math::mean(tensor, axes_vec);
             } else {
               throw std::runtime_error(
                   py::str("Invalid data type for axes. Got type '{}'")
                       .format(py::type::of(axes)));
             }
           });
}

}  // namespace jetdl
