#include "jetdl/math/reduction.h"

#include <vector>

#include "jetdl/math/kernel.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"
#include "jetdl/utils/reduction.h"

namespace jetdl {

Tensor _math_total_sum(const Tensor& a) {
  auto result_data = std::make_shared<std::vector<float>>(1);
  c_total_sum_cpu((*result_data).data(), (*a._data).data(), a.size);

  return Tensor(result_data, {}, a.requires_grad);
}

Tensor _math_sum_over_axes(const Tensor& a, const std::vector<size_t>& axes) {
  const std::vector<size_t>& result_shape = utils::get_shape(a.shape, axes);

  const size_t result_size = utils::get_size(result_shape);
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  const std::vector<size_t>& result_strides = utils::get_strides(result_shape);

  const std::vector<size_t>& dest_strides =
      utils::get_dest_strides(a.shape, result_strides, axes);

  const std::vector<size_t>& dest_idxs_vec = utils::populate_linear_idxs(
      a.shape, dest_strides, utils::OpType::REDUCTION);

  c_sum_over_axes_cpu(result_data->data(), a._data->data(),
                      dest_idxs_vec.data(), a.size);

  return Tensor(result_data, result_shape, a.requires_grad);
}

Tensor _math_sum_to_shape(const Tensor& tensor,
                          const std::vector<size_t>& shape) {
  if (tensor.shape == shape) {
    return tensor;
  }
  const std::vector<size_t>& axes =
      utils::get_broadcasted_axes(tensor.shape, shape);
  return _math_sum_over_axes(tensor, axes);
}

}  // namespace jetdl
