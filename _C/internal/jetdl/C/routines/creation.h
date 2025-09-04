#ifndef ROUTINES_CREATION_ONES_H
#define ROUTINES_CREATION_ONES_H

#include "jetdl/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor *c_routines_ones(const size_t *input_shape, const size_t ndim);

#ifdef __cplusplus
}
#endif

#endif
