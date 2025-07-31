#pragma once

#include <memory>
#include <vector>

namespace utils {

    struct IntPtrs {
        std::unique_ptr<int[]> ptr1;
        std::unique_ptr<int[]> ptr2;
    }; 

    inline int factorCeilingFunc(const int CURRENT, const int FACTOR) {
        // Rounds CURRENT to the first FACTOR multiple greater than CURRENT
        return ((CURRENT + FACTOR - 1) / FACTOR) * FACTOR;
    }

    std::unique_ptr<int[]> populateLinearIdxs(std::vector<int> shape, int* strides, const int offset);
    
}