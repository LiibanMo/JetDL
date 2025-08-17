#include "metadata.hpp"

#include <numeric>

namespace utils {

    namespace metadata {

        std::shared_ptr<float[]> flatten_nested_pylist(py::list& data) {
            std::vector<float> flat_vector;
            std::function<void(py::list)> flatten = 
                [&](py::list l) {
                for (auto item : l) {
                    if (py::isinstance<py::list>(item)) {
                        flatten(py::cast<py::list>(item));
                    } else {
                        if (!py::isinstance<py::int_>(item) && !py::isinstance<py::float_>(item)) {
                            auto input_type = item.get_type();
                            py::gil_scoped_acquire acquire;
                            throw py::type_error(
                                py::str("new(): invalid input type {}").format(input_type)
                            );
                        }
                        flat_vector.push_back(py::cast<float>(item));
                    }
                }
            };
            flatten(data);
            
            std::shared_ptr<float[]> result(new float[flat_vector.size()]);
            std::copy(flat_vector.begin(), flat_vector.end(), result.get());
            return result;
        }

        std::vector<int> get_shape(py::list& data) {
            std::vector<int> shape;
            if (data.empty()) {
                return shape;
            }   
            shape.push_back(static_cast<int>(data.size()));
            if (!data.empty() && py::isinstance<py::list>(data[0])) {
                py::list nested_list = py::cast<py::list>(data[0]);
                std::vector<int> nested_shape = get_shape(nested_list);
                shape.insert(shape.end(), nested_shape.begin(), nested_shape.end());
            }
            return shape;
        }

        const int get_ndim(const std::vector<int>& shape) {
            return shape.size();
        }

        std::vector<int> get_strides(const std::vector<int>& shape) {
            if (shape.empty()) {
                return std::vector<int>{};
            }
            const int ndim = shape.size();
            std::vector<int> strides (ndim, 1);
            for (int i = ndim-2; i >= 0; i--) {
                strides[i] = strides[i+1] * shape[i+1];
            }
            return strides;
        }

        const int get_size(const std::vector<int>& shape) {
            const int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            return size;
        }

        void assign_basic_metadata(Tensor& mutable_tensor, const std::vector<int>& shape) {
            mutable_tensor.shape = shape;
            mutable_tensor.ndim = get_ndim(shape);
            mutable_tensor.size = get_size(shape);
            mutable_tensor.strides = get_strides(shape);
        }

    } 

}