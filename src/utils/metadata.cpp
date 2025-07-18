#include "metadata.hpp"

#include <cstdlib>

namespace utils {

    namespace metadata {

        std::vector<float> flattenNestedPylist(py::list data) {
            std::vector<float> result;
            for (auto item : data) {
                if (py::isinstance<py::list>(item)) {
                    std::vector<float> nested_result = flattenNestedPylist(py::cast<py::list>(item));
                    result.insert(result.end(), nested_result.begin(), nested_result.end());
                } else {
                    result.push_back(py::cast<float>(item));
                }
            }
            return result;
        }

        std::vector<int> getShape(py::list data) {
            std::vector<int> shape;
            if (data.empty()) {
                return shape;
            }   
            shape.push_back(static_cast<int>(data.size()));
            if (!data.empty() && py::isinstance<py::list>(data[0])) {
                std::vector<int> nested_shape = getShape(py::cast<py::list>(data[0]));
                shape.insert(shape.end(), nested_shape.begin(), nested_shape.end());
            }
            return shape;
        }

        int getNumDim(const std::vector<int>& shape) {
            return shape.size();
        }

        std::vector<int> getStrides(const std::vector<int>& shape) {
            const int ndim = shape.size();
            std::vector<int> strides (ndim, 1);
            for (int i = ndim-2; i >= 0; i--) {
                strides[i] = strides[i+1] * shape[i+1];
            }
            return strides;
        }

        int getSize(const std::vector<int>& shape) {
            int size = 1;
            for (int i = 0; i < shape.size(); i++) {
                size *= shape[i];
            }
            return size;
        }

    } 

}