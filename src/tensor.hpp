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

inline void _obtain_strides(Tensor& tensor) {
    if (tensor.shape.empty()) {
        throw std::runtime_error("no shape allocated for strides to be obtained from.");
    }
    if (!tensor.ndim) {
        throw std::runtime_error("no ndim provided for strides to be calculated.");
    }
    tensor.strides = std::vector(tensor.ndim, 1);
    for (int idx = tensor.ndim-1; idx > 0; idx--) {
        tensor.strides[idx-1] = tensor.strides[idx] * tensor.shape[idx];
    }
}

#endif