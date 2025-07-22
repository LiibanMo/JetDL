#pragma once

#include <vector>

namespace utils {

    struct IntPtrs {
        int* ptr1;
        int* ptr2;
    }; 

    inline int factorCeilingFunc(const int CURRENT, const int FACTOR) {
        // Rounds CURRENT to the first FACTOR multiple greater than CURRENT
        return ((CURRENT + FACTOR - 1) / FACTOR) * FACTOR;
    }

    int* populateLinearIdxs(std::vector<int> shape, int* strides, const int offset);
    
}