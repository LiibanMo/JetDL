#include "jetdl/utils/py.h"

#include <memory>
#include <stdexcept>

#include "jetdl/utils/metadata.h"

namespace jetdl {
namespace utils {

void py_check_data_consistency_step(const py::list& data) {
  if (data.size() <= 1) {
    return;
  }
  const auto first_item = data[0];
  size_t first_item_size = 0;
  if (py::isinstance<py::list>(first_item)) {
    first_item_size = py::cast<py::list>(first_item).size();
  } else if (jetdl::utils::py_is_num(first_item)) {
    first_item_size = 0;
  } else {
    throw std::runtime_error(jetdl::utils::py_dtype_error_message(first_item));
  }

  for (size_t i = 1; i < data.size(); i++) {
    const auto current_item = data[i];
    size_t current_item_size = 0;
    if (jetdl::utils::py_is_list(current_item)) {
      current_item_size = py::cast<py::list>(current_item).size();
    } else if (jetdl::utils::py_is_num(current_item)) {
      current_item_size = 0;
    } else {
      throw std::runtime_error(
          jetdl::utils::py_dtype_error_message(first_item));
    }

    if (current_item_size != first_item_size) {
      throw py::value_error(py::str("input tensor must be consistent; mismatch "
                                    "in sub-list sizes. (expected {}, got {})")
                                .format(first_item_size, current_item_size));
    }
  }
}

void py_check_data_consistency(const py::list& data) {
  jetdl::utils::py_check_data_consistency_step(data);
  for (const auto& item : data) {
    if (py::isinstance<py::list>(item)) {
      py_check_data_consistency(py::cast<py::list>(item));
    }
  }
}

size_t py_get_ndim(const py::object& data) {
  if (jetdl::utils::py_is_num(data)) {
    return 0;
  } else if (!jetdl::utils::py_is_list(data)) {
    throw std::runtime_error(jetdl::utils::py_dtype_error_message(data));
  }

  size_t ndim = 1;
  auto item = py::cast<py::list>(data)[0];
  while (jetdl::utils::py_is_list(item)) {
    ndim++;
    py::list current_list = py::cast<py::list>(item);
    if (jetdl::utils::py_is_list(current_list[0])) {
      item = current_list[0];
    } else if (jetdl::utils::py_is_num(current_list[0])) {
      break;
    } else {
      throw std::runtime_error(
          jetdl::utils::py_dtype_error_message(current_list[0]));
    }
  }

  return ndim;
}

std::vector<size_t> py_get_shape(const py::object& data, const size_t ndim) {
  if (ndim == 0) {
    return {};
  }

  auto shape = std::vector<size_t>(ndim, 0);

  auto item = data;
  for (size_t i = 0; i < ndim; i++) {
    py::list current_list = py::cast<py::list>(item);
    shape[i] = (size_t)current_list.size();

    if (current_list.empty()) {
      break;
    } else if (jetdl::utils::py_is_list(current_list[0])) {
      item = current_list[0];
    }
  }

  return shape;
}

size_t py_get_size(const py::list& data) {
  size_t count = 0;
  for (const auto& item : data) {
    if (py_is_list(item)) {
      count += jetdl::utils::py_get_size(py::cast<py::list>(item));
    } else {
      count++;
    }
  }
  return count;
}

void py_flatten_list_to_vec(const py::list& data,
                            std::shared_ptr<float[]>& data_ptr, size_t& idx) {
  for (const auto& item : data) {
    if (jetdl::utils::py_is_list(item)) {
      py_flatten_list_to_vec(py::cast<py::list>(item), data_ptr, idx);
    } else if (jetdl::utils::py_is_num(item)) {
      data_ptr[idx] = py::cast<float>(item);
      idx++;
    } else {
      throw py::type_error(py::str("init: incorrect data type present in input "
                                   "data; contains type {}.")
                               .format(py::type::of(item)));
    }
  }
}

std::shared_ptr<float[]> py_flatten_list(const py::list& data) {
  const size_t size = py_get_size(data);

  auto data_ptr = std::shared_ptr<float[]>(new float[size]());

  size_t idx = 0;
  jetdl::utils::py_flatten_list_to_vec(data, data_ptr, idx);

  return data_ptr;
}

}  // namespace utils
}  // namespace jetdl
