#include "bindings.hpp"
#include "math.hpp"

namespace math {    

    void bind_add(py::module_& m) {
        m.def("c_add", &math::add, py::call_guard<py::gil_scoped_release>());
    }
    
    void bind_sub(py::module_& m) {
        m.def("c_sub", &math::sub, py::call_guard<py::gil_scoped_release>());
    }
    
    void bind_mul(py::module_& m) {
        m.def("c_mul", &math::mul, py::call_guard<py::gil_scoped_release>());
    }
    
    void bind_div(py::module_& m) {
        m.def("c_div", &math::div, py::call_guard<py::gil_scoped_release>());
    }
    
}