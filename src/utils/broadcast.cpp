#include "broadcast.hpp"
#include "auxillary.hpp"

namespace utils {

    namespace broadcast {

        IntPtrs BroadcastingUtilsObject::getBroadcastStrides() {
            const int ndim1 = this->ndim1;
            const int ndim2 = this->ndim2;
            const int max_ndim = this->max_ndim;
            
            IntPtrs stridesPtrs;
            stridesPtrs.ptr1 = (int*)std::calloc(max_ndim, sizeof(int)); 
            stridesPtrs.ptr2 = (int*)std::calloc(max_ndim, sizeof(int)); 
            if (!stridesPtrs.ptr1 || !stridesPtrs.ptr2) {
                throw std::runtime_error("Memory allocation failed.\n");
            }
            
            int strideA = 1, strideB = 1;
            int offset = 0;
            if (matmul) {
                strideA = this->shape1[ndim1-1];
                strideB = this->shape2[ndim2-1];
                offset = 2;
            }

            for (int i = max_ndim-offset-1; i >= 0; i--) {
                const int idx1 = i - max_ndim + ndim1;
                const int idx2 = i - max_ndim + ndim2;
                
                const int dim1 = (idx1 < 0) ? 1 : this->shape1[idx1];
                const int dim2 = (idx2 < 0) ? 1 : this->shape2[idx2];
                
                strideA *= this->shape1[idx1+1];
                strideB *= this->shape2[idx2+1];
                
                stridesPtrs.ptr1[i] = (dim1 < dim2 && dim1 <= 1) ? 0 : strideA;
                stridesPtrs.ptr2[i] = (dim2 < dim1 && dim2 == 1) ? 0 : strideB;
            }

            return stridesPtrs;
        }

        std::vector<int> BroadcastingUtilsObject::getResultShape() {
            const int ndim1 = this->ndim1;
            const int ndim2 = this->ndim2;
            const int max_ndim = this->max_ndim;

            std::vector<int> result_shape (max_ndim, 0);

            int offset = 0;
            if (this->matmul) {
                result_shape[max_ndim-2] = this->shape1[ndim1-2];
                result_shape[max_ndim-1] = this->shape2[ndim2-1];
                offset = 2;
            }

            for (int i = max_ndim-offset-1; i >= 0; i--) {
                const int idx1 = i - max_ndim + ndim1;
                const int idx2 = i - max_ndim + ndim2;

                const int dim1 = (idx1 < 0) ? 1 : this->shape1[idx1];
                const int dim2 = (idx2 < 0) ? 1 : this->shape2[idx2];

                result_shape[i] = std::max(dim1, dim2);
            }

            // Assumes only one operand can be a vector
            if (ndim1 == 1) {
                result_shape.erase(result_shape.end() - 2);
            } else if (ndim2 == 1) {
                result_shape.erase(result_shape.end() - 1);
            }
            
            return result_shape;
        }

    }

}