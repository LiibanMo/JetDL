#include "bindings.hpp"
#include "tensor.hpp"

#include <pybind11/stl.h>

namespace py = pybind11;

void bind_tensor_class(py::module_& m) {
    py::class_<Tensor>(m, "TensorBase")
        .def(py::init<py::list&, bool>(),
            py::arg("data"),
            py::arg("requires_grad")
        )
        .def(py::init<float, bool>(),
            py::arg("data"),
            py::arg("requires_grad")
        )
        .def_property_readonly("_data", [](Tensor& self) {
            std::vector<float> _data = std::vector<float>(self.size, 0.0f);
            std::copy(self._data.get(), self._data.get() + self.size, _data.begin());
            return _data;
        })
        .def_property_readonly("shape", [](const Tensor& self) {
            return py::tuple(py::cast(self.shape));
        })
        .def_readonly("ndim", &Tensor::ndim)
        .def_readonly("size", &Tensor::size)
        .def_property_readonly("strides", [](const Tensor& self) {
            return py::tuple(py::cast(self.strides));
        })

        .def_readonly("requires_grad", &Tensor::requires_grad)
        .def_readonly("grad_fn", &Tensor::grad_fn)
        .def_readonly("grad", &Tensor::grad)

        .def_readonly("is_contiguous", &Tensor::is_contiguous)
        .def_readonly("is_leaf", &Tensor::is_leaf);
}