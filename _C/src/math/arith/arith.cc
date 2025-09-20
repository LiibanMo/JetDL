#include "jetdl/math/arith.h"

#include <cstring>
#include <memory>
#include <vector>

#include "jetdl/autograd/math.h"
#include "jetdl/math/kernel.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

using ArithType = jetdl::utils::ArithType;

Tensor _math_ops(const Tensor& a, const Tensor& b, ArithType arith_type) {
  const std::vector<size_t>& shape =
      utils::get_result_shape(a.shape, b.shape, utils::OpType::ARITHMETIC);

  const auto& strides_pair =
      utils::get_strides(a.shape, b.shape, utils::OpType::ARITHMETIC);
  const std::vector<size_t>& stridesA = strides_pair.first;
  const std::vector<size_t>& stridesB = strides_pair.second;

  const std::vector<size_t>& idxsA =
      utils::populate_linear_idxs(shape, stridesA, utils::OpType::ARITHMETIC);
  const std::vector<size_t>& idxsB =
      utils::populate_linear_idxs(shape, stridesB, utils::OpType::ARITHMETIC);

  const size_t NA = a.shape[a.ndim - 1];
  const size_t NB = b.shape[b.ndim - 1];
  const size_t N = std::max(NA, NB);

  const size_t DATA_VEC_SIZE = utils::get_next_multiple(N, BLOCK_N_COLS);

  float* result_vec = new float[DATA_VEC_SIZE]();
  float* data1_vec = new float[DATA_VEC_SIZE]();
  float* data2_vec = new float[DATA_VEC_SIZE]();

  const size_t result_size = utils::get_size(shape);
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  void (*kernel)(const float*, const float*, float*, const size_t);
  if (arith_type == ArithType::ADD) kernel = c_add_cpu;
  if (arith_type == ArithType::SUB) kernel = c_sub_cpu;
  if (arith_type == ArithType::MUL) kernel = c_mul_cpu;
  if (arith_type == ArithType::DIV) kernel = c_div_cpu;

  const size_t total_num_rows = result_size / shape.back();
  if (NA == NB) {
    for (size_t row = 0; row < total_num_rows; row++) {
      std::copy(a._data->begin() + idxsA[row],
                a._data->begin() + idxsA[row] + N, data1_vec);
      std::copy(b._data->begin() + idxsB[row],
                b._data->begin() + idxsB[row] + N, data2_vec);
      kernel(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
      std::copy(result_vec, result_vec + N, result_data->begin() + row * N);
    }
  } else if (NA < NB && NA == 1) {
    for (size_t row = 0; row < total_num_rows; row++) {
      std::fill_n(data1_vec, N, a._data->at(idxsA[row]));
      std::copy(b._data->begin() + idxsB[row],
                b._data->begin() + idxsB[row] + N, data2_vec);
      kernel(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
      std::copy(result_vec, result_vec + N, result_data->begin() + row * N);
    }
  } else if (NA > NB && NB == 1) {
    for (size_t row = 0; row < total_num_rows; row++) {
      std::copy(a._data->begin() + idxsA[row],
                a._data->begin() + idxsA[row] + N, data1_vec);
      std::fill_n(data2_vec, N, b._data->at(idxsB[row]));
      kernel(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
      std::copy(result_vec, result_vec + N, result_data->begin() + row * N);
    }
  }

  delete[] result_vec;
  delete[] data1_vec;
  delete[] data2_vec;

  auto result_tensor =
      Tensor(result_data, shape, a.requires_grad || b.requires_grad);

  if (result_tensor.requires_grad) {
    if (arith_type == ArithType::ADD) {
      auto a_ptr = std::make_shared<Tensor>(a);
      auto b_ptr = std::make_shared<Tensor>(b);
      result_tensor.grad_fn = std::make_shared<AddBackward>(a_ptr, b_ptr);
    }
  }

  return result_tensor;
}

Tensor _math_ops_a_scalar(const Tensor& a, const Tensor& b,
                          ArithType arith_type) {
  const std::vector<size_t>& shape = b.shape;

  auto result_data = std::make_shared<std::vector<float>>(b.size);

  void (*kernel)(const float*, const float*, float*, const size_t);
  if (arith_type == ArithType::ADD) kernel = c_add_a_scalar_cpu;
  if (arith_type == ArithType::SUB) kernel = c_sub_a_scalar_cpu;
  if (arith_type == ArithType::MUL) kernel = c_mul_a_scalar_cpu;
  if (arith_type == ArithType::DIV) kernel = c_div_a_scalar_cpu;

  kernel(a._data->data(), b._data->data(), result_data->data(), b.size);
  return Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}

Tensor _math_ops_b_scalar(const Tensor& a, const Tensor& b,
                          ArithType arith_type) {
  const std::vector<size_t>& shape = a.shape;

  auto result_data = std::make_shared<std::vector<float>>(a.size);

  void (*kernel)(const float*, const float*, float*, const size_t);
  if (arith_type == ArithType::ADD) kernel = c_add_b_scalar_cpu;
  if (arith_type == ArithType::SUB) kernel = c_sub_b_scalar_cpu;
  if (arith_type == ArithType::MUL) kernel = c_mul_b_scalar_cpu;
  if (arith_type == ArithType::DIV) kernel = c_div_b_scalar_cpu;

  kernel(a._data->data(), b._data->data(), result_data->data(), a.size);

  return Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}

Tensor _math_ops_scalars(const Tensor& a, const Tensor& b,
                         ArithType arith_type) {
  auto result_data = std::make_shared<std::vector<float>>(1);

  void (*kernel)(const float*, const float*, float*);
  if (arith_type == ArithType::ADD) kernel = c_add_scalars_cpu;
  if (arith_type == ArithType::SUB) kernel = c_sub_scalars_cpu;
  if (arith_type == ArithType::MUL) kernel = c_mul_scalars_cpu;
  if (arith_type == ArithType::DIV) kernel = c_div_scalars_cpu;

  kernel(a._data->data(), b._data->data(), result_data->data());

  return Tensor(result_data, {}, a.requires_grad || b.requires_grad);
}

}  // namespace jetdl
