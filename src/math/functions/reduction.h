#ifndef MATH_FUNCTION_REDUCTION_H
#define MATH_FUNCTION_REDUCTION_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

    Tensor* c_math_total_sum(const Tensor* a);
    Tensor* c_math_sum_over_axes(const Tensor* a, const int* axes, const size_t naxes);

#ifdef __cplusplus
}
#endif
#endif