#ifndef JETDL_PYTHON_TENSOR_METHODS_H
#define JETDL_PYTHON_TENSOR_METHODS_H

#include <pybind11/pybind11.h>

#include "jetdl/tensor.h"

namespace py = pybind11;

namespace jetdl {

void bind_tensor_autograd_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor);
void bind_tensor_math_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor);
void bind_tensor_linalg_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor);
void bind_tensor_routines_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor);

}  // namespace jetdl

#endif
