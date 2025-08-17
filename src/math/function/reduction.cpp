#include "reduction.hpp"
#include "kernel.hpp"

#include "utils/auxiliary.hpp"
#include "utils/check.hpp"
#include "utils/metadata.hpp"
#include "utils/reduction.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

Tensor c_total_sum(const Tensor& tensor) {
    Tensor result_tensor = Tensor();

    // ----- Assigning metadata -----
    utils::metadata::assign_basic_metadata(result_tensor, {});
    result_tensor.requires_grad = tensor.requires_grad;
    // ------------------------------

    result_tensor._data = std::make_shared<float[]>(result_tensor.size);

    c_total_sum_cpu(tensor._data.get(), result_tensor._data.get(), tensor.size);

    return result_tensor;
}

Tensor c_sum_w_axes(const Tensor& tensor, const std::vector<int>& input_axes) {
    utils::check::axis_conditions(tensor.shape, input_axes);
    const std::vector<int> axes = utils::make_axes_positive(input_axes, tensor.ndim);

    Tensor result_tensor = Tensor();

    // ----- Assigning metadata -----
    utils::metadata::assign_basic_metadata(result_tensor, utils::reduction::get_shape(tensor.shape, axes));
    result_tensor.requires_grad = tensor.requires_grad;
    // ------------------------------

    result_tensor._data = std::make_shared<float[]>(result_tensor.size);

    std::vector<int> strides_mapped = utils::reduction::get_strides_for_calculation(tensor.shape, result_tensor.strides, axes);

    std::unique_ptr<int[]> dest_idxs = utils::populate_linear_idxs(tensor.shape, strides_mapped.data(), 0);
    
    c_sum_cpu(tensor._data.get(), result_tensor._data.get(), dest_idxs.get(), tensor.size);

    return result_tensor;
}

Tensor c_sum_to_size(const Tensor& tensor, const std::vector<int>& shape) { 
    if (tensor.ndim < shape.size()) {
        throw std::runtime_error("cannot sum to a size greater than the original tensor.");
    }

    utils::check::ops_broadcast_conditions(tensor.shape, shape);

    std::vector<int> axes = std::vector<int> ();

    for (int i = tensor.ndim-1; i >= 0; i--) {
        const int shape_idx = i - tensor.ndim + shape.size();

        if (shape[shape_idx] != tensor.shape[i] && shape[shape_idx] == 1) {
            axes.push_back(tensor.shape[i]);
        }
    }

    return c_sum_w_axes(tensor, axes);
}