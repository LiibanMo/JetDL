#include "bindings.hpp"

#include "creation.hpp"

#include <pybind11/stl.h>

namespace routines {

    void bind_submodule(py::module_& m) {
        m.def("c_ones", creation::ones);
    }

}