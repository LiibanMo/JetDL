#include "auxillary.hpp"

#include <algorithm>
#include <functional>
#include <numeric>

namespace utils {

    std::unique_ptr<int[]> populateLinearIdxs(std::vector<int> shape, int* strides, const int offset) {
        const int ndim = shape.size();
        const int size = std::accumulate(shape.begin(), shape.end() - offset, 1, std::multiplies<int>());
        
        std::unique_ptr<int[]> max_dim_values = std::make_unique<int[]>(ndim);

        std::transform(shape.begin(), shape.end() - offset, &max_dim_values[0], [](int x){return x - 1;});

        std::unique_ptr<int[]> lin_idxs = std::make_unique<int[]>(size);
        std::unique_ptr<int[]> idx = std::make_unique<int[]>(ndim);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < ndim; j++) {
                lin_idxs[i] += strides[j] * idx[j];
            }
            if (std::equal(idx.get(), idx.get() + ndim, max_dim_values.get())) {
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
        return lin_idxs;
    }

}