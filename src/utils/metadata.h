#ifndef UTILS_METADATA_H
#define UTILS_METADATA_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

    size_t utils_metadata_get_size(const size_t* shape, const size_t ndim);
    size_t* utils_metadata_get_strides(const size_t* shape, const size_t ndim);
    void utils_metadata_assign_basics(Tensor* mutable_tensor, const size_t* shape, const size_t ndim);

#ifdef __cplusplus    
}
#endif

#endif