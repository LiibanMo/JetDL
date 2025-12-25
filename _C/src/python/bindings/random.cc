#include "jetdl/random.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "jetdl/device.h"
#include "jetdl/python/random/bindings.h"

namespace py = pybind11;

namespace jetdl {

void bind_random_submodule(py::module_& m) {
  py::module_ random = m.def_submodule("random");

  random.def(
      "c_uniform",
      [](const float low, const float high, const std::vector<size_t>& shape,
         const size_t seed, const std::string& device_str) {
        return random::uniform(low, high, shape, seed,
                               Device::parse(device_str));
      },
      py::arg("low"), py::arg("high"), py::arg("shape"), py::arg("seed") = 123,
      py::arg("device") = "cpu");

  random.def(
      "c_normal",
      [](const float mean, const float std, const std::vector<size_t>& shape,
         const size_t seed, const std::string& device_str) {
        return random::normal(mean, std, shape, seed,
                              Device::parse(device_str));
      },
      py::arg("mean"), py::arg("std"), py::arg("shape"), py::arg("seed") = 123,
      py::arg("device") = "cpu");
}

}  // namespace jetdl
