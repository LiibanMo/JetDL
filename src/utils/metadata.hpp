#pragma once

#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace utils {

    namespace metadata {
        std::vector<float> flattenNestedPylist(py::list data);
        std::vector<int> getShape(py::list data);
        int getNumDim(const std::vector<int>& shape);
        std::vector<int> getStrides(const std::vector<int>& shape);
        int getSize(const std::vector<int>& shape);
    }

}