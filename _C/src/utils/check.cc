#include "jetdl/utils/check.h"

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <vector>

#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/metadata.h"

namespace py = pybind11;

namespace jetdl {
namespace utils {

void check_axes(const std::vector<size_t>& shape, const std::vector<int>& axes,
                SubModule submodule) {
  const int lower_limit_offset = (submodule == SubModule::ROUTINES) ? 1 : 0;
  const int upper_limit_offset = (submodule == SubModule::MATH) ? 1 : 0;

  const int lower_limit = -static_cast<int>(shape.size()) - lower_limit_offset;
  const int upper_limit = static_cast<int>(shape.size()) - upper_limit_offset;

  for (const int axis : axes) {
    if (axis > upper_limit || axis < lower_limit) {
      py::gil_scoped_acquire acquire;
      throw py::index_error(
          py::str(
              "dimension out of range (got {}, which is outside of [{},{}])")
              .format(axis, lower_limit, upper_limit));
    }
  }

  std::vector<size_t> updated_axes = make_axes_positive(axes, shape.size());

  for (size_t updated_axis : updated_axes) {
    if (std::count(updated_axes.begin(), updated_axes.end(), updated_axis) >
        1) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          py::str("dim {} appears multiple times in the list of axes")
              .format(updated_axis));
    }
  }
}

void check_ops_shapes(const std::vector<size_t>& shapeA,
                      const std::vector<size_t>& shapeB) {
  if (shapeA.size() != shapeB.size() ||
      !std::equal(shapeA.begin(), shapeA.end(), shapeB.begin())) {
    const size_t max_ndim = std::max(shapeA.size(), shapeB.size());

    for (int i = max_ndim - 1; i >= 0; i--) {
      const int idxA = i - static_cast<int>(max_ndim) + shapeA.size();
      const int idxB = i - static_cast<int>(max_ndim) + shapeB.size();

      const size_t dimA = (idxA < 0) ? 1 : shapeA[idxA];
      const size_t dimB = (idxB < 0) ? 1 : shapeB[idxB];

      if (dimA != dimB && dimA != 1 && dimB != 1) {
        py::gil_scoped_acquire acquire;
        throw py::value_error(
            py::str("operands could not be broadcasted together; incompatible "
                    "shapes."));
      }
    }
  }
}

void check_dot_shapes(const std::vector<size_t>& shapeA,
                      const std::vector<size_t>& shapeB) {
  if (!(shapeA.size() == 1 && shapeB.size() == 1)) {
    throw std::runtime_error("dot (C++): Wrong error checking used.");
  }

  if (shapeA[0] != shapeB[0]) {
    py::gil_scoped_acquire acquire;
    throw py::value_error(
        py::str("dot: Input operands have incompatible shapes; {} != {}")
            .format(shapeA[0], shapeB[0]));
  }
}

void check_vecmat_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB) {
  if (!(shapeA.size() == 1 && shapeB.size() > 1)) {
    throw std::runtime_error("vecmat (C++): Wrong error checking used.");
  }

  const size_t N = shapeA[0];
  if (shapeB[shapeB.size() - 2] != N) {
    py::gil_scoped_acquire acquire;
    throw py::value_error(
        py::str("matmul: Input operands have incompatible shapes; {} != {}")
            .format(N, shapeB[shapeB.size() - 2]));
  }
}

void check_matvec_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB) {
  if (!(shapeA.size() > 1 && shapeB.size() == 1)) {
    throw std::runtime_error("matvec (C++): Wrong error checking used.");
  }

  const size_t N = shapeB[0];

  if (shapeA[shapeA.size() - 1] != N) {
    py::gil_scoped_acquire acquire;
    throw py::value_error(
        py::str("matvec: Input operands have incompatible shapes; {} != {}")
            .format(shapeA[shapeA.size() - 1], N));
  }
}

void check_matmul_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB) {
  if (!(shapeA.size() >= 2 && shapeB.size() >= 2)) {
    throw std::runtime_error("matmul (C++): Wrong error checking used.");
  }

  if (shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2]) {
    py::gil_scoped_acquire acquire;
    throw py::value_error(
        py::str("matmul: Input operands have incompatible shapes; {} != {}")
            .format(shapeA[shapeA.size() - 1], shapeB[shapeB.size() - 2]));
  }

  const size_t max_ndim = std::max(shapeA.size(), shapeB.size());

  for (int i = max_ndim - 3; i >= 0; i--) {
    const int idxA = i - static_cast<int>(max_ndim) + shapeA.size();
    const int idxB = i - static_cast<int>(max_ndim) + shapeB.size();

    const size_t dimA = (idxA < 0) ? 1 : shapeA[idxA];
    const size_t dimB = (idxB < 0) ? 1 : shapeB[idxB];

    if (dimA != dimB && dimA != 1 && dimB != 1) {
      py::gil_scoped_acquire acquire;
      throw py::value_error(
          py::str("operands could not be broadcasted together along batch "
                  "dimensions"));
    }
  }
}

void check_reshape_shape(const std::vector<int>& new_shape, const size_t size) {
  size_t infer_count = 0;
  for (size_t i = 0; i < new_shape.size(); i++) {
    const int dim = new_shape[i];
    if (dim < 0 && dim != -1) {
      throw std::runtime_error(
          py::str("invalid shape dimension {} at index {} of {}")
              .format(i, dim, new_shape));
    }
    infer_count += (dim == -1) ? 1 : 0;
    if (infer_count > 1) {
      throw std::runtime_error(py::str("only one dimension can be inferred."));
    }
  }

  const int new_size = utils::get_size(new_shape);

  if (new_size != size) {
    throw std::runtime_error(
        py::str("shape '{}' is invalid for input for size {}")
            .format(new_shape, size));
  }
}

}  // namespace utils
}  // namespace jetdl
