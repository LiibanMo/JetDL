#include "jetdl/math.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "jetdl/python/math/bindings.h"

namespace py = pybind11;

void bind_math_submodule(py::module_& m) {
  py::module_ math = m.def_submodule("math");
  math.def("c_add", &jetdl::math::add);
  math.def("c_sub", &jetdl::math::sub);
  math.def("c_mul", &jetdl::math::mul);
  math.def("c_div", &jetdl::math::div);

  math.def("c_sum", [](const jetdl::Tensor& tensor, const py::object& axes) {
    if (py::isinstance<py::int_>(axes)) {
      const auto& axes_vec = std::vector<int>(1, py::cast<int>(axes));
      return jetdl::math::sum(tensor, axes_vec);
    } else if (py::isinstance<py::list>(axes) ||
               py::isinstance<py::tuple>(axes)) {
      const auto& axes_vec = axes.cast<std::vector<int>>();
      return jetdl::math::sum(tensor, axes_vec);
    } else if (axes.is_none()) {
      return jetdl::math::sum(tensor, {});
    } else {
      throw std::runtime_error(
          py::str("Invalid data type for axes. Got type '{}'")
              .format(py::type::of(axes)));
    }
  });
}
