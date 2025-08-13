#include "reduction.hpp"

#include <algorithm>
#include <set>
#include <vector>


namespace utils {

    namespace reduction {

        std::vector<int> get_shape(const std::vector<int>& shape, const std::vector<int>& axes) {
            std::vector<int> result_shape = std::vector<int>(shape.size(), 0);

            std::copy(shape.begin(), shape.end(), result_shape.begin());

            for (int axis : axes) {
                result_shape[axis] = 0;
            }

            result_shape.erase(std::remove(result_shape.begin(), result_shape.end(), 0), result_shape.end());

            return result_shape;
        }

        std::vector<int> get_strides_for_calculation(const std::vector<int>& input_shape, const std::vector<int>& result_strides, const std::vector<int>& axes) {
            std::set<int> reduction_axes_set(axes.begin(), axes.end());

            std::vector<int> mapped_strides(input_shape.size(), 0);

            int result_stride_idx = 0;

            for (int i = 0; i < input_shape.size(); ++i) {
                if (reduction_axes_set.find(i) == reduction_axes_set.end()) {
                    mapped_strides[i] = result_strides[result_stride_idx];
                    result_stride_idx++;
                }
            }

            return mapped_strides;
        }
    }
    
}
