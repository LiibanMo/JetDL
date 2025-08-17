#include "creation.hpp"
#include "utils/metadata.hpp"
#include <memory>

namespace creation {

    Tensor ones(const std::vector<int>& shape, const bool requires_grad) {
        Tensor result_tensor = Tensor();
        
        // ----- Assigning metadata -----
        utils::metadata::assign_basic_metadata(result_tensor, shape);
        
        result_tensor.is_contiguous = true;
        result_tensor.is_leaf = true;
        
        result_tensor.requires_grad = requires_grad;
        result_tensor.grad_fn = nullptr;
        result_tensor.grad = nullptr;
        // ------------------------------

        result_tensor._data = std::make_shared<float[]>(result_tensor.size);
        for (int i = 0; i < result_tensor.size; i++) {
            result_tensor._data[i] = 1.0f;
        }
        
        return result_tensor;
    }

}