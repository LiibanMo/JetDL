#include "jetdl/math.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "jetdl/python/tensor/methods.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/py.h"
namespace py = pybind11;

namespace {

void bind_tensor_add_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def(
      "__add__", [](const jetdl::Tensor& self, const py::object& other) {
        if (py::isinstance<jetdl::Tensor>(other)) {
          return jetdl::math::add(self, py::cast<jetdl::Tensor>(other));
        } else if (jetdl::utils::py_is_num(other)) {
          return jetdl::math::add(self, jetdl::Tensor(other));
        } else {
          throw py::type_error(py::str("{} object cannot be interpreted "
                                       "as a valid numerical type")
                                   .format(py::type::of(other)));
        }
      });
}

void bind_tensor_radd_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def("__radd__", [](const jetdl::Tensor& self,
                               const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      return jetdl::math::add(jetdl::Tensor(other), self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_sub_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def(
      "__sub__", [](const jetdl::Tensor& self, const py::object& other) {
        if (py::isinstance<jetdl::Tensor>(other)) {
          return jetdl::math::sub(self, py::cast<jetdl::Tensor>(other));
        } else if (jetdl::utils::py_is_num(other)) {
          return jetdl::math::sub(self, jetdl::Tensor(other));
        } else {
          throw py::type_error(py::str("{} object cannot be interpreted "
                                       "as a valid numerical type")
                                   .format(py::type::of(other)));
        }
      });
}

void bind_tensor_rsub_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def("__rsub__", [](const jetdl::Tensor& self,
                               const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      return jetdl::math::sub(jetdl::Tensor(other), self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_mul_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def(
      "__mul__", [](const jetdl::Tensor& self, const py::object& other) {
        if (py::isinstance<jetdl::Tensor>(other)) {
          return jetdl::math::mul(self, py::cast<jetdl::Tensor>(other));
        } else if (jetdl::utils::py_is_num(other)) {
          return jetdl::math::mul(self, jetdl::Tensor(other));
        } else {
          throw py::type_error(py::str("{} object cannot be interpreted "
                                       "as a valid numerical type")
                                   .format(py::type::of(other)));
        }
      });
}

void bind_tensor_rmul_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def("__rmul__", [](const jetdl::Tensor& self,
                               const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      return jetdl::math::mul(jetdl::Tensor(other), self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_div_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def(
      "__truediv__", [](const jetdl::Tensor& self, const py::object& other) {
        if (py::isinstance<jetdl::Tensor>(other)) {
          return jetdl::math::div(self, py::cast<jetdl::Tensor>(other));
        } else if (jetdl::utils::py_is_num(other)) {
          return jetdl::math::div(self, jetdl::Tensor(other));
        } else {
          throw py::type_error(py::str("{} object cannot be interpreted "
                                       "as a valid numerical type")
                                   .format(py::type::of(other)));
        }
      });
}

void bind_tensor_rdiv_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def("__rtruediv__", [](const jetdl::Tensor& self,
                                   const py::object& other) {
    if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other)) {
      return jetdl::math::div(jetdl::Tensor(other), self);
    } else {
      throw py::type_error(py::str("{} object cannot be interpreted "
                                   "as a valid numerical type")
                               .format(py::type::of(other)));
    }
  });
}

void bind_tensor_sum_method(py::class_<jetdl::Tensor>& py_tensor) {
  py_tensor.def("sum", [](const jetdl::Tensor& self, const py::object& axes) {
    if (py::isinstance<py::list>(axes) || py::isinstance<py::tuple>(axes)) {
      const std::vector<int>& axes_vec = py::cast<std::vector<int>>(axes);
      return jetdl::math::sum(self, axes_vec);
    } else if (py::isinstance<py::int_>(axes)) {
      const std::vector<int>& axes_vec = std::vector<int>(py::cast<int>(axes));
      return jetdl::math::sum(self, axes_vec);
    } else {
      throw py::type_error(
          py::str("{} object cannot be interpreted as a valid set of axes")
              .format(py::type::of(axes)));
    }
  });
}

}  // namespace

void bind_tensor_math_methods(py::class_<jetdl::Tensor>& py_tensor) {
  bind_tensor_add_method(py_tensor);
  bind_tensor_radd_method(py_tensor);

  bind_tensor_sub_method(py_tensor);
  bind_tensor_rsub_method(py_tensor);

  bind_tensor_mul_method(py_tensor);
  bind_tensor_rmul_method(py_tensor);

  bind_tensor_div_method(py_tensor);
  bind_tensor_rdiv_method(py_tensor);

  bind_tensor_sum_method(py_tensor);
}
