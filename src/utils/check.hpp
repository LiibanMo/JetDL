#pragma once

#include <vector>

namespace utils {

    namespace check {

        void matvecConditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void matmulConditions(const std::vector<int>& shape1, const std::vector<int>& shape2);

    }
    
}