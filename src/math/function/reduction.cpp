#include "reduction.hpp"
#include "kernel.hpp"

#include "utils/auxiliary.hpp"
#include "utils/check.hpp"
#include "utils/metadata.hpp"
#include "utils/reduction.hpp"

#include <memory>
#include <vector>

Tensor c_total_sum(const Tensor& tensor) {
    Tensor result_tensor = Tensor();

    // ----- Assigning metadata -----
    result_tensor.shape = {};
    result_tensor.ndim = 0;
    result_tensor.size = 1;
    result_tensor.strides = {};
    result_tensor.requires_grad = tensor.requires_grad;
    result_tensor.is_contiguous = true;
    result_tensor.is_leaf = false;
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
    result_tensor.shape = utils::reduction::get_shape(tensor.shape, axes);
    result_tensor.ndim = utils::metadata::get_ndim(result_tensor.shape);
    result_tensor.size = utils::metadata::get_size(result_tensor.shape);
    result_tensor.strides = utils::metadata::get_strides(result_tensor.shape);
    result_tensor.requires_grad = tensor.requires_grad;
    result_tensor.is_contiguous = true;
    result_tensor.is_leaf = false;
    // ------------------------------

    result_tensor._data = std::make_shared<float[]>(result_tensor.size);

    std::vector<int> strides_mapped = utils::reduction::get_strides_for_calculation(tensor.shape, result_tensor.strides, axes);

    std::unique_ptr<int[]> dest_idxs = utils::populate_linear_idxs(tensor.shape, strides_mapped.data(), 0);
    
    c_sum_cpu(tensor._data.get(), result_tensor._data.get(), dest_idxs.get(), tensor.size);

    return result_tensor;
}