#include "jetdl/math.h"

#include <pybind11/pybind11.h>

#include "jetdl/python/math/bindings.h"

namespace py = pybind11;

void bind_math_submodule(py::module_& m) {
  py::module_ math = m.def_submodule("math");
  math.def("c_add", &jetdl::math::add);
  math.def("c_sub", &jetdl::math::sub);
  math.def("c_mul", &jetdl::math::mul);
  math.def("c_div", &jetdl::math::div);

  math.def("c_sum", &jetdl::math::sum);
}
