#include "add.hpp"
#include "kernel.hpp"
#include "../../utils/broadcast.hpp"
#include "../../utils/check.hpp"
#include "../../utils/metadata.hpp"
#include <vector>

Tensor c_add_broadcasted(const Tensor& a, const Tensor& b) {
    utils::check::opsBroadcastConditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();
    utils::broadcast::BroadcastingUtilsObject BroadcastUtils(a.shape, b.shape, false);

    // ----- Assigning metadata -----
    result_tensor.shape = BroadcastUtils.getResultShape();
    result_tensor.ndim = utils::metadata::getNumDim(result_tensor.shape);
    result_tensor.size = utils::metadata::getSize(result_tensor.shape);
    result_tensor.strides = utils::metadata::getStrides(result_tensor.shape);
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    // ------------------------------

    utils::IntPtrs strides = BroadcastUtils.getBroadcastStrides();
    int* stridesA = strides.ptr1;
    int* stridesB = strides.ptr2;

    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);

    c_add_cpu(
        a._data.data(), b._data.data(), result_tensor._data.data(), 
        a.shape.data(), b.shape.data(), stridesA, stridesB, result_tensor.size, a.ndim, b.ndim
    );

    free(stridesA);
    free(stridesB);

    return result_tensor;
}