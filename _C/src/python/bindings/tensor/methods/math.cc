#include "jetdl/math.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "jetdl/python/tensor/methods.h"
#include "jetdl/tensor.h"

namespace py = pybind11;

namespace jetdl {

namespace {

using TensorOpsTensor = std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>&,
                                                    std::shared_ptr<Tensor>&);
using TensorOpsPyObject = std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>&,
                                                      py::object& other);

void bind_tensor_add_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__add__", static_cast<TensorOpsTensor>(&math::add),
                py::is_operator());
  py_tensor.def("__add__",
                static_cast<TensorOpsPyObject>(
                    [](std::shared_ptr<Tensor>& self, py::object& other) {
                      auto operand_tensor = std::make_shared<Tensor>(other);
                      return math::add(self, operand_tensor);
                    }),
                py::is_operator());
}

void bind_tensor_radd_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__radd__", [](std::shared_ptr<Tensor> self,
                               const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      auto operand_tensor = std::make_shared<Tensor>(other);
      return math::add(operand_tensor, self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_sub_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__sub__", static_cast<TensorOpsTensor>(&math::sub),
                py::is_operator());
  py_tensor.def("__sub__",
                static_cast<TensorOpsPyObject>(
                    [](std::shared_ptr<Tensor>& self, py::object& other) {
                      auto operand_tensor = std::make_shared<Tensor>(other);
                      return math::sub(self, operand_tensor);
                    }),
                py::is_operator());
}

void bind_tensor_rsub_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__rsub__", [](std::shared_ptr<Tensor> self,
                               const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      auto operand_tensor = std::make_shared<Tensor>(other);
      return math::sub(operand_tensor, self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_mul_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__mul__", static_cast<TensorOpsTensor>(&math::mul),
                py::is_operator());
  py_tensor.def("__mul__",
                static_cast<TensorOpsPyObject>(
                    [](std::shared_ptr<Tensor>& self, py::object& other) {
                      auto operand_tensor = std::make_shared<Tensor>(other);
                      return math::mul(self, operand_tensor);
                    }),
                py::is_operator());
}

void bind_tensor_rmul_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__rmul__", [](std::shared_ptr<Tensor> self,
                               const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      auto operand_tensor = std::make_shared<Tensor>(other);
      return math::mul(operand_tensor, self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_div_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__truediv__", static_cast<TensorOpsTensor>(&math::div),
                py::is_operator());
  py_tensor.def("__truediv__",
                static_cast<TensorOpsPyObject>(
                    [](std::shared_ptr<Tensor>& self, py::object& other) {
                      auto operand_tensor = std::make_shared<Tensor>(other);
                      return math::div(self, operand_tensor);
                    }),
                py::is_operator());
}

void bind_tensor_rdiv_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__rtruediv__", [](std::shared_ptr<Tensor> self,
                                   const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      auto operand_tensor = std::make_shared<Tensor>(other);
      return math::div(operand_tensor, self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_neg_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("__neg__", &math::neg);
}

void bind_tensor_pow_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def(
      "__pow__", [](std::shared_ptr<Tensor>& self, py::object& exponent) {
        if (py::isinstance<py::int_>(exponent) ||
            py::isinstance<py::float_>(exponent)) {
          return math::pow(self, py::cast<float>(exponent));
        } else {
          throw py::type_error(
              py::str("can only raise power by int or float. Got type '{}'")
                  .format(py::type::of(exponent)));
        }
      });
}

void bind_tensor_sum_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def(
      "sum",
      [](std::shared_ptr<Tensor> self, const py::object& axes) {
        if (py::isinstance<py::list>(axes) || py::isinstance<py::tuple>(axes)) {
          std::vector<int> axes_vec = py::cast<std::vector<int>>(axes);
          return math::sum(self, axes_vec);
        } else if (py::isinstance<py::int_>(axes)) {
          std::vector<int> axes_vec = std::vector<int>{py::cast<int>(axes)};
          return math::sum(self, axes_vec);
        } else if (axes.is_none()) {
          return math::sum(self);
        } else {
          throw py::type_error(
              py::str("{} object cannot be interpreted as a valid set of axes")
                  .format(py::type::of(axes)));
        }
      },
      py::arg("axes") = py::none());
}

void bind_tensor_sum_to_shape_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("sum_to_shape", [](std::shared_ptr<Tensor>& self,
                                   const py::object& shape) {
    if (py::isinstance<py::list>(shape) || py::isinstance<py::tuple>(shape)) {
      auto shape_vec = py::cast<std::vector<size_t>>(shape);
      return math::sum_to_shape(self, shape_vec);
    } else {
      throw py::type_error(
          py::str("type '{}' for shape input for tensor reduction.")
              .format(py::type::of(shape)));
    }
  });
}

void bind_tensor_mean_method(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def(
      "mean",
      [](std::shared_ptr<Tensor> self, const py::object& axes) {
        if (py::isinstance<py::list>(axes) || py::isinstance<py::tuple>(axes)) {
          std::vector<int> axes_vec = py::cast<std::vector<int>>(axes);
          return math::mean(self, axes_vec);
        } else if (py::isinstance<py::int_>(axes)) {
          std::vector<int> axes_vec = std::vector<int>{py::cast<int>(axes)};
          return math::mean(self, axes_vec);
        } else if (axes.is_none()) {
          return math::mean(self);
        } else {
          throw py::type_error(
              py::str("{} object cannot be interpreted as a valid set of axes")
                  .format(py::type::of(axes)));
        }
      },
      py::arg("axes") = py::none());
}

void bind_tensor_transcendental_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  py_tensor.def("exp", &math::exp);
  py_tensor.def("log", &math::log);
  py_tensor.def("log10", &math::log10);
  py_tensor.def("log2", &math::log2);
  py_tensor.def("sin", &math::sin);
  py_tensor.def("cos", &math::cos);
  py_tensor.def("tanh", &math::tanh);
  py_tensor.def("sinh", &math::sinh);
  py_tensor.def("cosh", &math::cosh);
  py_tensor.def("abs", &math::abs);
  py_tensor.def("sign", &math::sign);
  py_tensor.def(
      "clamp",
      [](std::shared_ptr<Tensor>& self, float min_val, float max_val) {
        return math::clamp(self, min_val, max_val);
      },
      py::arg("min"), py::arg("max"));
}

}  // namespace

void bind_tensor_math_methods(
    py::class_<Tensor, std::shared_ptr<Tensor>>& py_tensor) {
  bind_tensor_add_method(py_tensor);
  bind_tensor_radd_method(py_tensor);

  bind_tensor_sub_method(py_tensor);
  bind_tensor_rsub_method(py_tensor);

  bind_tensor_mul_method(py_tensor);
  bind_tensor_rmul_method(py_tensor);

  bind_tensor_div_method(py_tensor);
  bind_tensor_rdiv_method(py_tensor);

  bind_tensor_neg_method(py_tensor);

  bind_tensor_pow_method(py_tensor);

  bind_tensor_sum_method(py_tensor);
  bind_tensor_sum_to_shape_method(py_tensor);

  bind_tensor_mean_method(py_tensor);

  bind_tensor_transcendental_methods(py_tensor);
}

}  // namespace jetdl
