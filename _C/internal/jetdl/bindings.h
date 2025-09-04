#ifndef JETDL_PYBIND11_H
#define JETDL_PYBIND11_H

#include "jetdl/tensor.h"
#include <pybind11/pybind11.h>

#ifdef __cplusplus

namespace py = pybind11;

struct TensorDeleter {
  void operator()(Tensor *tensor) const { destroy_tensor(tensor); }
};

Tensor *init_tensor(const py::object &data);

void bind_linalg_submodule(py::module_ &m);
void bind_math_submodule(py::module_ &m);
void bind_routines_submodule(py::module_ &m);
void bind_tensor_submodule(py::module_ &m);

#endif

#endif
