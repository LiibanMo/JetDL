#ifndef MATH_OPS_ARITH
#define MATH_OPS_ARITH

#include "tensor/tensor.h"
#include "utils/auxiliary.h"

#ifdef __cplusplus
extern "C" {
#endif

    Tensor* c_math_ops(const Tensor* a, const Tensor* b, ArithType arith_type);

#ifdef __cplusplus
}
#endif
#endif