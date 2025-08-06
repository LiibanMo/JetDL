#include "bindings.hpp"
#include "linalg.hpp"

namespace linalg {

    void bind_dot(py::module_& m) {
        m.def("c_dot", &linalg::dot, py::call_guard<py::gil_scoped_release>());
    }

    void bind_matmul(py::module_& m) {
        m.def("c_matmul", &linalg::matmul, py::call_guard<py::gil_scoped_release>());
    }

    void bind_T(py::module_& m) {
        m.def("c_transpose", &linalg::T, py::call_guard<py::gil_scoped_release>());
    }
    
    void bind_mT(py::module_& m) {
        m.def("c_matrix_transpose", &linalg::mT, py::call_guard<py::gil_scoped_release>());
    }

}