#include "jetdl/routines.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "jetdl/python/routines/bindings.h"
#include "jetdl/utils/check.h"

namespace py = pybind11;

namespace jetdl {

void bind_routines_submodule(py::module_& m) {
  py::module_ routines = m.def_submodule("routines");
  routines.def("c_zeros", &jetdl::zeros);
  routines.def("c_ones", &jetdl::ones);
  routines.def("c_fill", &jetdl::fill);
  routines.def("c_reshape", [](std::shared_ptr<Tensor>& tensor,
                               const py::object& shape) {
    if (!py::isinstance<py::list>(shape) && !py::isinstance<py::tuple>(shape)) {
      throw py::type_error(
          py::str("type '{}' not valid as shape input when reshaping tensor")
              .format(py::type::of(shape)));
    }

    const auto& shape_vec = py::cast<std::vector<int>>(shape);
    return jetdl::reshape(tensor, shape_vec);
  });

  routines.def(
      "c_squeeze", [](std::shared_ptr<Tensor>& input, const py::object& axes) {
        if (axes.is_none()) {
          return jetdl::squeeze(input, {});
        } else if (py::isinstance<py::list>(axes) ||
                   py::isinstance<py::tuple>(axes)) {
          return jetdl::squeeze(input, py::cast<std::vector<int>>(axes));
        } else if (py::isinstance<py::int_>(axes) && py::cast<int>(axes) >= 0) {
          const auto& input_axis = std::vector<int>{py::cast<int>(axes)};
          return jetdl::squeeze(input, input_axis);
        } else {
          throw py::type_error(
              py::str("type '{}' not valid as axes input when squeezing tensor")
                  .format(py::type::of(axes)));
        }
      });

  routines.def("c_unsqueeze", [](std::shared_ptr<Tensor>& input,
                                 const py::object& axis) {
    if (!py::isinstance<py::int_>(axis)) {
      throw py::type_error(
          py::str("type '{}' not valid as axes input when unsqueezing tensor")
              .format(py::type::of(axis)));
    }

    const int input_axis = py::cast<int>(axis);
    const int lower_limit = -static_cast<int>(input->ndim);
    const int upper_limit = static_cast<int>(input->ndim);

    if (input_axis < lower_limit || input_axis > upper_limit) {
      throw std::runtime_error(
          py::str("{} not in range [{}, {}] for unsqueezing")
              .format(input_axis, lower_limit, upper_limit));
    }
    return jetdl::unsqueeze(input, input_axis);
  });

  routines.def("c_contiguous", &jetdl::contiguous);
}

}  // namespace jetdl
