#include "auxillary.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace utils {

    int* populateLinearIdxs(std::vector<int> shape, int* strides, const int offset) {
        const int ndim = shape.size();
        const int size = std::accumulate(shape.begin(), shape.end() - offset, 1, std::multiplies<int>());
        
        int* max_dim_values = (int*)malloc(shape.size() * sizeof(int));
        if (!max_dim_values) {
            throw std::runtime_error("Memory allocation failed.\n");
        }
        std::transform(shape.begin(), shape.end() - offset, &max_dim_values[0], [](int x){return x - 1;});

        int* lin_idxs = (int*)malloc(size * sizeof(int));
        int* idx = (int*)malloc(ndim * sizeof(int));
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