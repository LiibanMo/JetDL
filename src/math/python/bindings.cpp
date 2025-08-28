#include "bindings.h"
#include "math/math.h"

#include <pybind11/stl.h>

void bind_math_submodule(py::module_& m) {
    py::module_ math = m.def_submodule("math");
    math.def("c_ops", &math_ops);
    math.def("c_sum", &math_sum);
}