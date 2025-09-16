#ifndef JETDL_UTILS_PY_HPP
#define JETDL_UTILS_PY_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jetdl {
namespace utils {

template <typename T>
inline bool py_is_list(const T& data) {
  return py::isinstance<py::list>(data);
}

template <typename T>
inline bool py_is_num(const T& data) {
  return py::isinstance<py::int_>(data) || py::isinstance<py::float_>(data);
}

inline py::str py_dtype_error_message(const py::object& item) {
  return py::str("could not infer dtype of {}").format(py::type::of(item));
}

void py_check_data_consistency_step(const py::list& data);
void py_check_data_consistency(const py::list& data);

size_t py_get_ndim(const py::object& data);
std::vector<size_t> py_get_shape(const py::object& data, const size_t ndim);

size_t py_get_size(const py::list& data);
void py_flatten_list_to_vec(const py::list& data, std::vector<float>& vec);
std::shared_ptr<std::vector<float>> py_flatten_list(const py::list& data);

}  // namespace utils
}  // namespace jetdl
#endif
