#ifndef JETDL_UTILS_CHECK_HPP
#define JETDL_UTILS_CHECK_HPP

#include <pybind11/pybind11.h>

#include <vector>

#include "jetdl/utils/auxiliary.h"

namespace py = pybind11;

class Tensor;

namespace jetdl {
namespace utils {

template <typename T>
inline void check_shape_dims(const std::vector<T>& shape) {
  for (const T dim : shape) {
    if (dim <= 0) {
      throw py::value_error(
          py::str("non-positive dimensions are not allowed.\n"));
    }
  }
}

void check_axes(const std::vector<size_t>& shape, const std::vector<int>& axes,
                SubModule submodule = SubModule::MATH);

void check_ops_shapes(const std::vector<size_t>& shapeA,
                      const std::vector<size_t>& shapeB);

void check_dot_shapes(const std::vector<size_t>& shapeA,
                      const std::vector<size_t>& shapeB);

void check_vecmat_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB);

void check_matvec_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB);

void check_matmul_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB);

void check_reshape_shape(const std::vector<int>& new_shape, const size_t size);

}  // namespace utils
}  // namespace jetdl

#endif
