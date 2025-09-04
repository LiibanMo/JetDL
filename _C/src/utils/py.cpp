#include "jetdl/utils/py.h"
#include <stdexcept>

void python_utils_check_data_consistency_step(const py::list &data) {
  if (data.size() <= 1) {
    return;
  }

  const auto first_item = data[0];
  size_t first_item_size;
  if (python_utils_is_pylist(first_item)) {
    first_item_size = py::cast<py::list>(first_item).size();
  } else if (python_utils_is_num(first_item)) {
    first_item_size = 0;
  } else {
    throw std::runtime_error(python_utils_dtype_error_message(first_item));
  }

  for (size_t i = 1; i < data.size(); i++) {
    const auto current_item = data[i];

    size_t current_item_size;
    if (python_utils_is_pylist(current_item)) {
      current_item_size = py::cast<py::list>(current_item).size();
    } else if (python_utils_is_num(current_item)) {
      current_item_size = 0;
    } else {
      throw std::runtime_error(python_utils_dtype_error_message(first_item));
    }

    if (current_item_size != first_item_size) {
      throw py::value_error(py::str("input tensor must be consistent; mismatch "
                                    "in sub-list sizes. (expected {}, got {})")
                                .format(first_item_size, current_item_size));
    }
  }
}

void python_utils_check_data_consistency(const py::list &data) {
  python_utils_check_data_consistency_step(data);
  for (const auto &item : data) {
    if (python_utils_is_pylist(item)) {
      python_utils_check_data_consistency(py::cast<py::list>(item));
    }
  }
}

size_t python_utils_get_ndim(const py::object &data) {
  if (python_utils_is_num(data)) {
    return 0;
  } else if (!python_utils_is_pylist(data)) {
    throw std::runtime_error(python_utils_dtype_error_message(data));
  }

  size_t ndim = 1;
  auto item = py::cast<py::list>(data)[0];
  while (python_utils_is_pylist(item)) {
    ndim++;
    py::list current_list = py::cast<py::list>(item);
    if (python_utils_is_pylist(current_list[0])) {
      item = current_list[0];
    } else if (python_utils_is_num(current_list[0])) {
      break;
    } else {
      throw std::runtime_error(
          python_utils_dtype_error_message(current_list[0]));
    }
  }

  return ndim;
}

size_t *python_utils_get_shape(const py::object &data, const size_t ndim) {
  if (ndim == 0) {
    return (size_t *)calloc(0, sizeof(size_t));
  }

  size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
  if (!shape) {
    throw std::bad_alloc();
  }

  auto item = data;
  for (size_t i = 0; i < ndim; i++) {
    py::list current_list = py::cast<py::list>(item);
    shape[i] = (size_t)current_list.size();

    if (current_list.empty()) {
      break;
    } else if (python_utils_is_pylist(current_list[0])) {
      item = current_list[0];
    }
  }

  return shape;
}

size_t python_utils_get_size(const py::list &data) {
  size_t count = 0;
  for (const auto &item : data) {
    if (python_utils_is_pylist(item)) {
      count += python_utils_get_size(py::cast<py::list>(item));
    } else {
      count++;
    }
  }
  return count;
}

void python_utils_populate_ptr(const py::list &data, float *&ptr) {
  for (const auto &item : data) {
    if (python_utils_is_pylist(item)) {
      python_utils_populate_ptr(py::cast<py::list>(item), ptr);
    } else if (python_utils_is_num(item)) {
      *ptr = py::cast<float>(item);
      ++ptr;
    } else {
      py::gil_scoped_acquire acquire;
      throw py::type_error(py::str("init: incorrect data type present in input "
                                   "data; contains type {}.")
                               .format(py::type::of(item)));
    }
  }
}

float *python_utils_flatten_list(const py::list &data) {
  const size_t size = python_utils_get_size(data);

  float *_data = (float *)malloc(size * sizeof(float));
  if (!_data) {
    throw std::bad_alloc();
  }

  float *ptr = _data;
  python_utils_populate_ptr(data, ptr);

  return _data;
}
