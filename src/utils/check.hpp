#pragma once

#include <vector>

namespace utils {

    namespace check {

        void opsBroadcastConditions(const std::vector<int> shape1, const std::vector<int> shape2);
        void dotConditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void vecmatConditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void matvecConditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void matmulConditions(const std::vector<int>& shape1, const std::vector<int>& shape2);

    }
    
}