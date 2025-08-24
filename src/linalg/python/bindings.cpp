#include "bindings.h"
#include "linalg/product/matmul.h"

void bind_linalg_submodule(py::module_& m) {
    py::module_ linalg = m.def_submodule("linalg");

    linalg.def("c_dot", 
        &c_linalg_dot, 
        py::call_guard<py::gil_scoped_release>()
    );
}