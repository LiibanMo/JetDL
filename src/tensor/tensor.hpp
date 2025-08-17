#pragma once

#include "autograd/function.hpp"

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

class Tensor {
    public:
        // basic metadata {
        std::shared_ptr<float[]> _data;
        std::vector<int> shape;
        int ndim;
        int size;
        std::vector<int> strides;
        // }

        bool is_contiguous;
        bool is_leaf;

        bool requires_grad;
        std::shared_ptr<Function> grad_fn;
        std::shared_ptr<Tensor> grad;

        Tensor(py::list& data, bool requires_grad);
        Tensor(const float data, bool requires_grad);
        Tensor();
        ~Tensor() = default;

        Tensor copy();
};