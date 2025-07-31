#pragma once

#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace utils {

    namespace metadata {
        std::shared_ptr<float[]> flattenNestedPylist(py::list data);
        std::vector<int> getShape(py::list data);
        const int getNumDim(const std::vector<int>& shape);
        std::vector<int> getStrides(const std::vector<int>& shape);
        const int getSize(const std::vector<int>& shape);
    }

}