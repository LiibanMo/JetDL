#include "auxillary.hpp"

#include <algorithm>
#include <cstdlib>

namespace utils {

    int* populateLinearIdxs(int* max_dim_values, int* strides, const int ndim, const int size) {
        int* lin_idxs = (int*)std::calloc(size, sizeof(int));
        int* idx = (int*)std::calloc(ndim, sizeof(int));
        if (!lin_idxs || !idx) {
            std::runtime_error("Memory allocation failed.\n");
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < ndim; j++) {
                lin_idxs[i] += strides[j] * idx[j];
            }
            if (std::equal(idx, idx + ndim, max_dim_values)) {
                break;
            }
            for (int axis = ndim-1; axis >= 0; axis--) {
                idx[axis]++;
                if (idx[axis] <= max_dim_values[axis]) {
                    break;
                }
                idx[axis] = 0;
            }
        }
        std::free(idx);
        return lin_idxs;
    }
    
}