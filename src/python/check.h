#pragma once

#include <stddef.h>

void utils_check_axes(const size_t* shape, const size_t ndim, const int* axes, const size_t axes_ndim);
void utils_check_ops_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB);
void utils_check_dot_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB);
void utils_check_vecmat_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB);
void utils_check_matvec_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB);
void utils_check_matmul_shapes(const size_t* shapeA, const size_t ndimA, const size_t* shapeB, const size_t ndimB);