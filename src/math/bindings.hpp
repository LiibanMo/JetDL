#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace math {    
    namespace ops {
        void bind_submodule(py::module_& m);
    }
    namespace function {
        void bind_submodule(py::module_& m);
    }
}