#include "tensor/python/bindings.h"

#include "tensor/python/utils.h"
#include "tensor/tensor.h"

#include <memory>
#include <new>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

Tensor *init_tensor(const py::object &data) {
  float *_data = NULL;
  if (python_utils_is_num(data)) {
    _data = (float *)malloc(sizeof(float));
    if (!_data)
      throw std::bad_alloc();
    *_data = py::cast<float>(data);
    return create_tensor(_data, NULL, 0);

  } else if (python_utils_is_pylist(data)) {
    python_utils_check_data_consistency(data);
    _data = python_utils_flatten_list(data);
    size_t ndim = python_utils_get_ndim(data);
    size_t *shape = python_utils_get_shape(data, ndim);
    return create_tensor(_data, shape, ndim);

  } else {
    throw std::runtime_error(python_utils_dtype_error_message(data));
  }
}

void bind_tensor_submodule(py::module_ &m) {
  py::class_<Tensor, std::unique_ptr<Tensor, TensorDeleter>>(m, "TensorBase")
      .def(py::init([](const py::object &data) {
        Tensor *tensor_ptr = init_tensor(data);
        if (!tensor_ptr)
          throw std::bad_alloc();
        return std::unique_ptr<Tensor, TensorDeleter>(tensor_ptr);
      }))
      .def_property_readonly(
          "_data",
          [](Tensor &self) {
            std::vector<float> _data = std::vector<float>(self.size, 0.0f);
            std::copy(self._data, self._data + self.size, _data.begin());
            return _data;
          })
      .def_property_readonly("shape",
                             [](Tensor &self) {
                               std::vector<size_t> shape_vec =
                                   python_utils_ptr_to_vec(self.shape,
                                                           self.ndim);
                               return py::tuple(py::cast(shape_vec));
                             })
      .def_readonly("ndim", &Tensor::ndim)
      .def_readonly("size", &Tensor::size)
      .def_property_readonly("strides",
                             [](Tensor &self) {
                               std::vector<size_t> strides_vec =
                                   python_utils_ptr_to_vec(self.strides,
                                                           self.ndim);
                               return py::tuple(py::cast(strides_vec));
                             })
      .def_readonly("is_contiguous", &Tensor::is_contiguous);

  m.def("c_destroy_tensor", &destroy_tensor);
}
