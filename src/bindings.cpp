#include "tensor/tensor.hpp"
#include "linalg/linalg.hpp"
#include "math/math.hpp"

#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_Cpp, m) {
    py::class_<Tensor>(m, "TensorBase")
        .def(py::init<py::list, bool>(),
            py::arg("data"),
            py::arg("requires_grad") = false
        )
        .def(py::init<float, bool>(),
            py::arg("data"),
            py::arg("requires_grad") = false
        )
        .def_property_readonly("_data", [](Tensor& self) {
            std::vector<float> _data = std::vector<float>(self.size, 0.0f);
            std::copy(self._data.get(), self._data.get() + self.size, _data.begin());
            return _data;
        })
        .def_property_readonly("shape", [](const Tensor& self){
            return py::tuple(py::cast(self.shape));
        })
        .def_readonly("ndim", &Tensor::ndim)
        .def_readonly("size", &Tensor::size)
        .def_property_readonly("strides", [](const Tensor& self){
            return py::tuple(py::cast(self.strides));
        })
        .def_readonly("requires_grad", &Tensor::requires_grad)
        .def_readonly("is_contiguous", &Tensor::is_contiguous);
       
    m.def("c_add", &math::add, py::call_guard<py::gil_scoped_release>());
    m.def("c_sub", &math::sub, py::call_guard<py::gil_scoped_release>());
    m.def("c_mul", &math::mul, py::call_guard<py::gil_scoped_release>());
    m.def("c_div", &math::div, py::call_guard<py::gil_scoped_release>());
    
    m.def("c_dot", &linalg::dot, py::call_guard<py::gil_scoped_release>());
    m.def("c_matmul", &linalg::matmul, py::call_guard<py::gil_scoped_release>());
    m.def("c_transpose", &linalg::T, py::call_guard<py::gil_scoped_release>());
    m.def("c_matrix_transpose", &linalg::mT, py::call_guard<py::gil_scoped_release>());
}