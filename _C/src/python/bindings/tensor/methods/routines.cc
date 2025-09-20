#include "jetdl/routines.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace jetdl {

void bind_tensor_routines_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("reshape", &jetdl::reshape);
}

}  // namespace jetdl
