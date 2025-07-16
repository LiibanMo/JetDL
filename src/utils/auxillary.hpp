#pragma once

namespace utils {

    struct IntPtrs {
        int* ptr1;
        int* ptr2;
    }; 
    
    int* populateLinearIdxs(int* max_dim_values, int* strides, const int ndim, const int size);

}