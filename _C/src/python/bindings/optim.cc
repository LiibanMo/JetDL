#include "jetdl/optim/optim.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "jetdl/python/optim/bindings.h"
#include "jetdl/tensor.h"

namespace py = pybind11;

namespace jetdl {

namespace {

void bind_optim_zero_grad(py::module_& m) {
  m.def("c_zero_grad", [](std::vector<std::shared_ptr<Tensor>>& params) {
    for (auto& param : params) {
      param->grad = nullptr;
    }
  });
}

void bind_optim_sgd_step(py::module_& m) { m.def("c_sgd", &optim::sgd_step); }

}  // namespace

void bind_optim_submodule(py::module_& m) {
  py::module_ optim = m.def_submodule("optim");
  bind_optim_zero_grad(optim);
  bind_optim_sgd_step(optim);
}

}  // namespace jetdl