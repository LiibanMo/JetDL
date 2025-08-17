#include "bindings.hpp"
#include "autograd/graph/backward.hpp"

namespace autograd {
    void bind_submodule(py::module_& m) {
        m.def("c_backward", &autograd::backward, py::call_guard<py::gil_scoped_acquire>());
    }
}