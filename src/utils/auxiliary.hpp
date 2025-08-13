#pragma once

#include <memory>
#include <vector>

namespace utils {

    struct IntPtrs {
        std::unique_ptr<int[]> ptr1;
        std::unique_ptr<int[]> ptr2;
    }; 

    inline int factor_ceiling_func(const int current, const int factor) {
        // Rounds CURRENT to the first FACTOR multiple greater than CURRENT
        return ((current + factor - 1) / factor) * factor;
    }

    std::unique_ptr<int[]> populate_linear_idxs(std::vector<int> shape, int* strides, const int offset);
    
    std::vector<int> make_axes_positive(const std::vector<int>& axes, const int ndim);
    
}