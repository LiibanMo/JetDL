#pragma once

#include "tensor/tensor.hpp"

#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace utils {

    namespace metadata {

        std::shared_ptr<float[]> flatten_nested_pylist(py::list& data);
        std::vector<int> get_shape(py::list& data);
        const int get_ndim(const std::vector<int>& shape);
        std::vector<int> get_strides(const std::vector<int>& shape);
        const int get_size(const std::vector<int>& shape);

    }

}