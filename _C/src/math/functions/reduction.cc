#include "jetdl/math/reduction.h"

#include <vector>

#include "jetdl/math/kernel.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/metadata.h"
#include "jetdl/utils/reduction.h"

jetdl::Tensor _math_total_sum(const jetdl::Tensor& a) {
  auto result_data = std::make_shared<std::vector<float>>(1);
  c_total_sum_cpu((*result_data).data(), (*a._data).data(), a.size);

  return jetdl::Tensor(result_data, {}, a.requires_grad);
}

jetdl::Tensor _math_sum_over_axes(const jetdl::Tensor& a,
                                  const std::vector<int>& axes) {
  std::vector<size_t> updated_axes =
      jetdl::utils::make_axes_positive(axes, a.ndim);
  std::sort(updated_axes.begin(), updated_axes.end());

  const std::vector<size_t>& result_shape =
      jetdl::utils::get_shape(a.shape, updated_axes);

  const size_t result_size = jetdl::utils::get_size(result_shape);
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  const std::vector<size_t>& result_strides =
      jetdl::utils::get_strides(result_shape);

  const std::vector<size_t>& dest_strides =
      jetdl::utils::get_dest_strides(a.shape, result_strides, updated_axes);

  const std::vector<size_t>& dest_idxs_vec = jetdl::utils::populate_linear_idxs(
      a.shape, dest_strides, jetdl::utils::OpType::REDUCTION);

  c_sum_over_axes_cpu(result_data->data(), a._data->data(),
                      dest_idxs_vec.data(), a.size);

  return jetdl::Tensor(result_data, result_shape, a.requires_grad);
}
