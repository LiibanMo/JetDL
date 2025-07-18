#pragma once

namespace utils {

    struct IntPtrs {
        int* ptr1;
        int* ptr2;
    }; 

    inline int factorCeilingFunc(const int CURRENT, const int FACTOR) {
        // Rounds CURRENT to the first FACTOR multiple greater than CURRENT
        return ((CURRENT + FACTOR - 1) / FACTOR) * FACTOR;
    }

    int* populateLinearIdxs(int* max_dim_values, int* strides, const int ndim, const int size);

}