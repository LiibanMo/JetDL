#include "jetdl/math.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "jetdl/python/math/bindings.h"

namespace py = pybind11;

namespace jetdl {

void bind_math_submodule(py::module_& m) {
  py::module_ math = m.def_submodule("math");
  math.def("c_add", &math::add);
  math.def("c_sub", &math::sub);
  math.def("c_mul", &math::mul);
  math.def("c_div", &math::div);

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
}

}  // namespace jetdl
