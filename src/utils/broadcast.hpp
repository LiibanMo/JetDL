#pragma once

#include "auxiliary.hpp"

namespace utils {

    namespace broadcast {
        
        class BroadcastingUtilsObject {
            public:
                std::vector<int> shape1;
                std::vector<int> shape2;
                int ndim1;
                int ndim2;
                int max_ndim;
                bool matmul; // if false, it broadcasts for general linalg ops (i.e. add, sub, etc.)

                inline BroadcastingUtilsObject(const std::vector<int>& shape1, const std::vector<int>& shape2, bool matmul) {
                    this->shape1 = shape1;
                    this->shape2 = shape2;
                    this->ndim1 = shape1.size();
                    this->ndim2 = shape2.size();
                    this->max_ndim = std::max(ndim1, ndim2);
                    this->matmul = matmul;
                };

                IntPtrs get_broadcast_strides();

                std::vector<int> get_result_shape();

        };
            
        static inline int get_batch_size(std::vector<int> shape) {
            int batch_size = 1;
            for (int i = shape.size()-3; i >= 0; i--) {
                batch_size *= shape[i];
            }
            return batch_size;
        }

    }

}