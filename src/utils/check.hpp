#pragma once

#include <vector>

namespace utils {

    namespace check {

        void axis_conditions(const std::vector<int>& shape, const std::vector<int>& axes);
        void ops_broadcast_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void dot_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void vecmat_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void matvec_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2);
        void matmul_conditions(const std::vector<int>& shape1, const std::vector<int>& shape2);

    }
    
}