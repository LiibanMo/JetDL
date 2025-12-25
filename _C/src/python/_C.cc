#include <pybind11/pybind11.h>

#include "jetdl/device.h"
#include "jetdl/python/linalg/bindings.h"
#include "jetdl/python/math/bindings.h"
#include "jetdl/python/nn/bindings.h"
#include "jetdl/python/optim/bindings.h"
#include "jetdl/python/random/bindings.h"
#include "jetdl/python/routines/bindings.h"
#include "jetdl/python/tensor/bindings.h"

PYBIND11_MODULE(_C, m) {
  jetdl::bind_tensor_submodule(m);
  jetdl::bind_linalg_submodule(m);
  jetdl::bind_math_submodule(m);
  jetdl::bind_random_submodule(m);
  jetdl::bind_routines_submodule(m);
  jetdl::bind_optim_submodule(m);
  jetdl::bind_nn_submodule(m);

  // CUDA utilities
  m.def("cuda_is_available", &jetdl::cuda_is_available,
        "Returns True if CUDA is available");
  m.def("cuda_device_count", &jetdl::cuda_device_count,
        "Returns the number of available CUDA devices");
}
