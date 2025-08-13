#pragma once

#include <vector>

namespace utils{
    namespace reduction {
        std::vector<int> get_shape(const std::vector<int>& shape, const std::vector<int>& axes);
        std::vector<int> get_strides_for_calculation(const std::vector<int>& input_shape, const std::vector<int>& result_strides, const std::vector<int>& axes);
    }
}