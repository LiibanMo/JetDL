#ifndef UTILS_REDUCTION_H
#define UTILS_REDUCTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

    size_t* utils_reduction_get_shape(
        const size_t* shape, const size_t ndim, const size_t* axes, const size_t naxes
    );
    
    size_t* utils_reduction_get_dest_strides(
        const size_t* original_shape, const size_t original_ndim, const size_t* result_strides,
        const size_t* axes, const size_t naxes
    ); 
    
#ifdef __cplusplus
}
#endif

#endif