#ifndef MATH_OPS_ARITH
#define MATH_OPS_ARITH

#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor *c_math_ops(const Tensor *a, const Tensor *b,
                   const ArithType arith_type);
Tensor *c_math_ops_a_scalar(const Tensor *a, const Tensor *b,
                            const ArithType arith_type);
Tensor *c_math_ops_b_scalar(const Tensor *a, const Tensor *b,
                            const ArithType arith_type);
Tensor *c_math_ops_scalars(const Tensor *a, const Tensor *b,
                           const ArithType arith_type);

#ifdef __cplusplus
}
#endif
#endif
