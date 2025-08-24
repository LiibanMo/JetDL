#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

template<typename T>
inline bool python_utils_is_pylist(const T& data) {
    return py::isinstance<py::list>(data);
}

template<typename T>
inline bool python_utils_is_num(const T& data) {
    return py::isinstance<py::int_>(data) || py::isinstance<py::float_>(data);
}

inline py::str python_utils_dtype_error_message(const py::object& item) {
    return py::str("could not infer dtype of {}").format(py::type::of(item));
}

void python_utils_check_data_consistency_step(const py::list& data);
void python_utils_check_data_consistency(const py::list& data);

size_t python_utils_get_ndim(const py::object& data);
size_t* python_utils_get_shape(const py::object& data, const size_t ndim);

size_t python_utils_get_size(const py::list& data);
void python_utils_populate_ptr(const py::list& data, float*& ptr);
float* python_utils_flatten_list(const py::list& data);

template<typename T>
std::vector<T> python_utils_ptr_to_vec(T* ptr, const size_t N) {
    if (!ptr) {
        return {};
    }
    std::vector<T> vec = std::vector<T>(N);
    std::copy(ptr, ptr + N, vec.begin());
    return vec;
}