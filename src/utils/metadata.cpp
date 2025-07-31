#include "metadata.hpp"

#include <numeric>

namespace utils {

    namespace metadata {

        std::shared_ptr<float[]> flattenNestedPylist(py::list data) {
            std::vector<float> flat_vector;
            std::function<void(py::list)> flatten = 
                [&](py::list l) {
                for (auto item : l) {
                    if (py::isinstance<py::list>(item)) {
                        flatten(py::cast<py::list>(item));
                    } else {
                        flat_vector.push_back(py::cast<float>(item));
                    }
                }
            };
            flatten(data);
            
            std::shared_ptr<float[]> result(new float[flat_vector.size()]);
            std::copy(flat_vector.begin(), flat_vector.end(), result.get());
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

        const int getNumDim(const std::vector<int>& shape) {
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

        const int getSize(const std::vector<int>& shape) {
            const int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            return size;
        }

    } 

}