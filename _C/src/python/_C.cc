#include <pybind11/pybind11.h>

#include "jetdl/python/linalg/bindings.h"
#include "jetdl/python/math/bindings.h"
#include "jetdl/python/routines/bindings.h"
#include "jetdl/python/tensor/bindings.h"

PYBIND11_MODULE(_C, m) {
  bind_tensor_submodule(m);
  bind_linalg_submodule(m);
  bind_math_submodule(m);
  bind_routines_submodule(m);
}
