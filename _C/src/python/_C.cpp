#include "jetdl/bindings.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(_C, m) {
  bind_tensor_submodule(m);
  bind_linalg_submodule(m);
  bind_math_submodule(m);
  bind_routines_submodule(m);
}
