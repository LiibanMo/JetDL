#include "broadcast.hpp"
#include "auxillary.hpp"
#include "metadata.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace utils {

    namespace broadcast {

        IntPtrs BroadcastingUtilsObject::getBroadcastStrides() {
            const int ndim1 = this->ndim1;
            const int ndim2 = this->ndim2;
            const int max_ndim = this->max_ndim;
            
            IntPtrs stridesPtrs;
            stridesPtrs.ptr1 = std::unique_ptr<int[]>(new int[max_ndim]());
            stridesPtrs.ptr2 = std::unique_ptr<int[]>(new int[max_ndim]());
            if (!stridesPtrs.ptr1 || !stridesPtrs.ptr2) {
                throw std::runtime_error("Memory allocation failed.\n");
            }
            
            std::vector<int> stridesA = utils::metadata::getStrides(shape1);
            std::vector<int> stridesB = utils::metadata::getStrides(shape2);

            const int offset = matmul ? 2 : 0;

            for (int i = max_ndim-offset-1; i >= 0; i--) {
                const int idx1 = i - max_ndim + ndim1;
                const int idx2 = i - max_ndim + ndim2;

                const int dim1 = (idx1 < 0) ? 1 : this->shape1[idx1];
                const int dim2 = (idx2 < 0) ? 1 : this->shape2[idx2];
                
                stridesPtrs.ptr1[i] = (dim1 == 1 && dim1 < dim2) ? 0 : stridesA[idx1];
                stridesPtrs.ptr2[i] = (dim2 == 1 && dim2 < dim1) ? 0 : stridesB[idx2];
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
            if (this->matmul) {
                if (ndim1 == 1) {
                    result_shape.erase(result_shape.end() - 2);
                } else if (ndim2 == 1) {
                    result_shape.erase(result_shape.end() - 1);
                }
            }
            
            return result_shape;
        }

    }

}