#include "check.h"
#include "utils/auxiliary.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void utils_check_axes(const size_t* shape, const size_t ndim, const int* axes, const size_t axes_ndim) {
    for (size_t i = 0; i < axes_ndim; i++) {
        const int axis = axes[i];

        py::gil_scoped_acquire acquire;

        if (axis >= ndim || axis < -ndim) {
            throw py::index_error(
                py::str(
                    "dimension out of range (got {}, which is outside of [{},{}])"
                )
                .format(axis, -ndim, ndim-1)
            );
        }
    }

    size_t* updated_axes = utils_make_axes_positive(axes, axes_ndim, ndim);
    for (size_t i = 0; i < ndim; i++) {
        size_t axis = updated_axes[i];

        const size_t freq = utils_get_count(updated_axes, &axis, ndim, sizeof(size_t));

        if (freq > 1) {
            throw std::runtime_error(
                py::str(
                    "dim {} appears multiple times in the list of axes"
                )
                .format(axis)
            );
        }
    }

    free(updated_axes);
    updated_axes = NULL;
}

void utils_check_ops_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB) {
    if (ndimA != ndimB || memcmp(shapeA, shapeB, ndimA)) {
        const size_t max_ndim = UTILS_GET_MAX(ndimA, ndimB);
        
        for (int i = max_ndim-1; i >= 0; i--) {
            const int idxA = i - (int)max_ndim + (int)ndimA;
            const int idxB = i - (int)max_ndim + (int)ndimB;
            
            const size_t dimA = (idxA < 0) ? 1 : shapeA[idxA];
            const size_t dimB = (idxB < 0) ? 1 : shapeB[idxB];
            
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                py::gil_scoped_acquire acquire;
                throw py::value_error(
                    py::str(
                        "operands could not be broadcasted together; incompatible shapes."
                    )
                );
            }
        }
    }
}

void utils_check_dot_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB) {
    if (!(ndimA == 1 && ndimB == 1)) {
        fprintf(stderr, "dot (C++): Wrong error checking used.");
        return;
    }

    if (shapeA[0] != shapeB[0]) {
        py::gil_scoped_acquire acquire;

        throw py::value_error(
            py::str(
                "dot: Input operands have incompatible shapes; {} != {}"
            )
            .format(shapeA[0], shapeB[0])
        );
    }
}

void utils_check_vecmat_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB) {
    // (N) @ (..., N, P)
    if (!(ndimA == 1 && ndimB > 1)) {
        fprintf(stderr, "vecmat (C++): Wrong error checking used.");
        return;
    }

    const size_t N = shapeA[0];
    if (shapeB[ndimB-2] != N) {
        py::gil_scoped_acquire acquire;
        throw py::value_error(
            py::str(
                "matmul: Input operands have incompatible shapes; {} != {}"
            )
            .format(N, shapeB[ndimB-2])
        );
    }
}

void utils_check_matvec_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB) {
    // (..., M, N) @ (N)
    if (!(ndimA > 1 && ndimB == 1)) {
        throw std::runtime_error("matvec (C++): Wrong error checking used.");
    } 
    
    const size_t N = shapeB[0];

    if (shapeA[ndimA-1] != N) {
        py::gil_scoped_acquire acquire;

        throw py::value_error(
            py::str(
                "matvec: Input operands have incompatible shapes; {} != {}"
            )
            .format(shapeA[ndimA-1], N)
        );
    }
}

void utils_check_matmul_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB) {
    // (..., M, N) @ (..., N, P)
    if (!(ndimA >= 2 && ndimB >= 2)) {
        fprintf(stderr, "matmul (C++): Wrong error checking used.");
        return;
    }
    
    if (shapeA[ndimA-1] != shapeB[ndimB-2]) {
        py::gil_scoped_acquire acquire;

        throw py::value_error(
            py::str(
                "matmul: Input operands have incompatible shapes; {} != {}"
            )
            .format(shapeA[ndimA-1], shapeB[ndimB-2])
        );
    }

    const size_t max_ndim = UTILS_GET_MAX(ndimA, ndimB);

    for (int i = max_ndim-3; i >= 0; i--) {
        const int idxA = i - (int)max_ndim + (int)ndimA;
        const int idxB = i - (int)max_ndim + (int)ndimB;

        const size_t dimA = (idxA < 0) ? 1 : shapeA[idxA];
        const size_t dimB = (idxB < 0) ? 1 : shapeB[idxB];

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            py::gil_scoped_acquire acquire;

            throw py::value_error(
                py::str("operands could not be broadcasted together along batch dimensions")
            );
        }
    } 
}
