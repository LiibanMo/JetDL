#include "jetdl/utils/reduction.h"

#include <memory>
#include <vector>

#include "jetdl/autograd/math.h"
#include "jetdl/math/function.h"
#include "jetdl/math/kernel.h"
#include "jetdl/routines.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> _math_total_sum(std::shared_ptr<Tensor>& a) {
  auto result_data = std::shared_ptr<float[]>(new float[1]());
  c_total_sum_cpu(result_data.get(), a->_data.get(), a->size);

  auto result_tensor = std::make_shared<Tensor>(
      result_data, std::vector<size_t>{}, a->requires_grad);

  return result_tensor;
}

std::shared_ptr<Tensor> _math_sum_over_axes(std::shared_ptr<Tensor>& a,
                                            const std::vector<size_t>& axes) {
  const std::vector<size_t>& result_shape = utils::get_shape(a->shape, axes);

  const size_t result_size = utils::get_size(result_shape);
  auto result_data = std::shared_ptr<float[]>(new float[result_size]());

  const std::vector<size_t>& result_strides = utils::get_strides(result_shape);

  const std::vector<size_t>& dest_strides =
      utils::get_dest_strides(a->shape, result_strides, axes);

  const std::vector<size_t>& dest_idxs_vec = utils::populate_linear_idxs(
      a->shape, dest_strides, utils::OpType::REDUCTION);

  c_sum_over_axes_cpu(result_data.get(), a->_data.get(), dest_idxs_vec.data(),
                      a->size);

  auto result_tensor =
      std::make_shared<Tensor>(result_data, result_shape, a->requires_grad);

  return result_tensor;
}

std::shared_ptr<Tensor> _math_sum_to_shape(std::shared_ptr<Tensor>& tensor,
                                           const std::vector<size_t>& shape) {
  if (tensor->shape == shape) {
    return tensor;
  }
  const std::vector<size_t>& axes =
      utils::get_broadcasted_axes(shape, tensor->shape);

  std::shared_ptr<Tensor> result_tensor = _math_sum_over_axes(tensor, axes);

  return view(result_tensor, shape);
}

}  // namespace jetdl
