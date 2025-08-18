#include "check.hpp"
#include "utils/auxiliary.hpp"

#include <algorithm>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace utils {

    namespace check {

        void axis_conditions(const std::vector<int>& shape, const std::vector<int>& axes) {
            const int ndim = shape.size();
            for (int axis : axes) {
                py::gil_scoped_acquire acquire;
                if (axis >= ndim || axis < -ndim) {
                    throw py::index_error(
                        py::str(
                            "dimension out of range (got {}, which is outside of [{},{}])"
                        )
                        .format(axis, -ndim, ndim-1)
                    );
                }
                const std::vector<int> updated_axes = utils::make_axes_positive(axes, ndim);
                const int freq = std::count(updated_axes.begin(), updated_axes.end(), axis);
                if (freq > 1) {
                    throw std::runtime_error(
                        py::str(
                            "dim {} appears multiple times in the list of axes"
                        )
                        .format(axis)
                    );
                }
            }

        }

        void ops_broadcast_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2) {
            if (!std::equal(shape1.begin(), shape2.end(), shape1.begin(), shape2.end())) {
                const int ndim1 = shape1.size();
                const int ndim2 = shape2.size();
                const int max_ndim = std::max(ndim1, ndim2);
                
                for (int i = max_ndim-1; i >= 0; i--) {
                    const int idx1 = i - max_ndim + ndim1;
                    const int idx2 = i - max_ndim + ndim2;
                    
                    const int dim1 = (idx1 < 0) ? 1 : shape1[idx1];
                    const int dim2 = (idx2 < 0) ? 1 : shape2[idx2];
                    
                    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                        py::gil_scoped_acquire acquire;
                        throw py::value_error(
                            py::str("operands could not be broadcasted together.")
                        );
                    }
                }
            }
        }

        void dot_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2) {
            // (N) @ (N)
            const int ndim1 = shape1.size();
            const int ndim2 = shape2.size();

            if (!(ndim1 == 1 && ndim2 == 1)) {
                throw std::runtime_error("dot (C++): Wrong error checking used.");
            }

            if (shape1[0] != shape2[0]) {
                py::gil_scoped_acquire acquire;
                throw py::value_error(
                    py::str(
                        "matmul: Input operands have incompatible shapes; {} != {}"
                    )
                    .format(shape1[0], shape2[0])
                );
            }
        }

        void matvec_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2) {
            // (..., M, N) @ (N)
            const int ndim1 = shape1.size(); 
            const int ndim2 = shape2.size(); // == 1

            if (!(ndim1 > 1 && ndim2 == 1)) {
                throw std::runtime_error("matvec (C++): Wrong error checking used.");
            } 
            
            const int N = shape2[0];
            if (shape1[ndim1-1] != N) {
                py::gil_scoped_acquire acquire;
                throw py::value_error(
                    py::str(
                        "matmul: Input operands have incompatible shapes; {} != {}"
                    )
                    .format(shape1[ndim1-1], N)
                );
            }
        }

        void vecmat_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2) {
            // (N) @ (..., N, P)
            const int ndim1 = shape1.size(); // == 1
            const int ndim2 = shape2.size();

            if (!(ndim1 == 1 && ndim2 > 1)) {
                throw std::runtime_error("vecmat (C++): Wrong error checking used.");
            }

            const int N = shape1[0];
            if (shape2[ndim2-2] != N) {
                py::gil_scoped_acquire acquire;
                throw py::value_error(
                    py::str(
                        "matmul: Input operands have incompatible shapes; {} != {}"
                    )
                );
            }
        }

        void matmul_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2) {
            // (..., M, N) @ (..., N, P)
            const int ndim1 = shape1.size();
            const int ndim2 = shape2.size();

            if (!(ndim1 >= 2 && ndim2 >= 2)) {
                throw std::runtime_error("matmul (C++): Wrong error checking used.");
            }
            
            if (shape1[ndim1-1] != shape2[ndim2-2]) {
                py::gil_scoped_acquire acquire;
                throw py::value_error(
                    py::str(
                        "matmul: Input operands have incompatible shapes; {} != {}"
                    )
                    .format(shape1[ndim1-1], shape2[ndim2-2])
                );
            }

            const int max_ndim = std::max(ndim1, ndim2);
            for (int i = max_ndim-3; i >= 0; i--) {
                const int idx1 = i - max_ndim + ndim1;
                const int idx2 = i - max_ndim + ndim2;

                const int dim1 = (idx1 < 0) ? 1 : shape1[idx1];
                const int dim2 = (idx2 < 0) ? 1 : shape2[idx2];

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    py::gil_scoped_acquire acquire;
                    throw py::value_error(
                        py::str("operands could not be broadcasted together along batch dimensions")
                    );
                }
            } 
        }

    }

}