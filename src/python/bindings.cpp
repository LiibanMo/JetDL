#include "bindings.h"
#include "utils/check.h"

#include <pybind11/stl.h>

void bind_utils_submodule(py::module_& m) {
    py::module_ utils = m.def_submodule("utils");

    utils.def("c_utils_check_axes", &utils_check_axes);
    utils.def("c_utils_check_ops_shapes", &utils_check_ops_shapes);
    utils.def("c_utils_check_dot_shapes", &utils_check_dot_shapes);
    utils.def("c_utils_check_vecmat_shapes", &utils_check_vecmat_shapes);
    utils.def("c_utils_check_matvec_shapes", &utils_check_matvec_shapes);
    utils.def("c_utils_check_matmul_shapes", &utils_check_matmul_shapes);
}