#include "bindings.hpp"
#include "math.hpp"

#include <pybind11/stl.h>

namespace math {    

    namespace ops {

        void bind_submodule(py::module_& m) {

            m.def("c_add", &math::ops::add, py::call_guard<py::gil_scoped_release>());
            
            m.def("c_sub", &math::ops::sub, py::call_guard<py::gil_scoped_release>());
            
            m.def("c_mul", &math::ops::mul, py::call_guard<py::gil_scoped_release>());
            
            m.def("c_div", &math::ops::div, py::call_guard<py::gil_scoped_release>());
        
        }

    }

    namespace function {

        void bind_submodule(py::module_& m) {
            m.def("c_total_sum", &math::function::total_sum, py::call_guard<py::gil_scoped_release>());
            m.def("c_sum_w_axes", &math::function::sum_w_axes, py::call_guard<py::gil_scoped_release>());
        }

    }
}