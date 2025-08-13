#include "bindings.hpp"
#include "linalg.hpp"

#include <pybind11/stl.h>

namespace linalg {

    void bind_submodule(py::module_& m) {

        m.def("c_dot", &linalg::dot, py::call_guard<py::gil_scoped_release>());
        
        m.def("c_matmul", &linalg::matmul, py::call_guard<py::gil_scoped_release>());
        
        m.def("c_transpose", &linalg::T, py::call_guard<py::gil_scoped_release>());
        
        m.def("c_matrix_transpose", &linalg::mT, py::call_guard<py::gil_scoped_release>());
        
    }

}