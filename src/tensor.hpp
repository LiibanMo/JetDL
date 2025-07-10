#pragma once

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

class Tensor {
    private: 
        std::vector<float> _flatten_nested_pylist(py::list data);
        std::vector<int> _obtain_shape(py::list data);

    public:
        Tensor(py::list data, bool requires_grad);
        Tensor();
        ~Tensor() = default;

        std::vector<float> _data;
        std::vector<int> shape;
        int ndim;
        int size;
        std::vector<int> strides;
        bool requires_grad;
};

#endif