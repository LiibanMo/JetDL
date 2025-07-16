#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

class Tensor {
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