#include "jetdl/math/arith.h"

#include <cstring>
#include <memory>
#include <vector>

#include "jetdl/math/kernel.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

jetdl::Tensor _math_ops(const jetdl::Tensor& a, const jetdl::Tensor& b,
                        jetdl::utils::ArithType arith_type) {
  const std::vector<size_t>& shape = jetdl::utils::get_result_shape(
      a.shape, b.shape, jetdl::utils::OpType::ARITHMETIC);

  auto strides_pair = jetdl::utils::get_strides(
      a.shape, b.shape, jetdl::utils::OpType::ARITHMETIC);
  const std::vector<size_t>& stridesA = strides_pair.first;
  const std::vector<size_t>& stridesB = strides_pair.second;

  const std::vector<size_t>& idxsA = jetdl::utils::populate_linear_idxs(
      shape, stridesA, jetdl::utils::OpType::ARITHMETIC);
  const std::vector<size_t>& idxsB = jetdl::utils::populate_linear_idxs(
      shape, stridesB, jetdl::utils::OpType::ARITHMETIC);

  const size_t NA = a.shape[a.ndim - 1];
  const size_t NB = b.shape[b.ndim - 1];
  const size_t N = std::max(NA, NB);

  const size_t DATA_VEC_SIZE = (N + 7) & ~7;  // Next multiple of 8

  float* result_vec = new float[DATA_VEC_SIZE]();
  float* data1_vec = new float[DATA_VEC_SIZE]();
  float* data2_vec = new float[DATA_VEC_SIZE]();

  const size_t result_size = jetdl::utils::get_size(shape);
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  void (*kernel)(const float*, const float*, float*, const size_t);
  if (arith_type == jetdl::utils::ArithType::ADD) kernel = c_add_cpu;
  if (arith_type == jetdl::utils::ArithType::SUB) kernel = c_sub_cpu;
  if (arith_type == jetdl::utils::ArithType::MUL) kernel = c_mul_cpu;
  if (arith_type == jetdl::utils::ArithType::DIV) kernel = c_div_cpu;

  const size_t total_num_rows = result_size / shape.back();

  if (NA == NB) {
    for (size_t row = 0; row < total_num_rows; row++) {
      std::copy(data1_vec, data1_vec + N, (*a._data).begin() + idxsA[row]);
      std::copy(data2_vec, data2_vec + N, (*b._data).begin() + idxsB[row]);
      kernel(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
      std::copy((*result_data).begin() + row * N,
                (*result_data).begin() + (row + 1) * N, result_vec);
    }
  } else if (NA < NB && NA == 1) {
    for (size_t row = 0; row < total_num_rows; row++) {
      std::fill_n(data1_vec, N, (*a._data)[0]);
      std::copy(data2_vec, data2_vec + N, (*b._data).begin() + idxsB[row]);
      kernel(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
      std::copy((*result_data).begin() + row * N,
                (*result_data).begin() + (row + 1) * N, result_vec);
    }
  } else if (NA > NB && NB == 1) {
    for (size_t row = 0; row < total_num_rows; row++) {
      std::copy(data1_vec, data1_vec + N, (*a._data).begin() + idxsA[row]);
      std::fill_n(data2_vec, N, (*b._data)[0]);
      kernel(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
      std::copy((*result_data).begin() + row * N,
                (*result_data).begin() + (row + 1) * N, result_vec);
    }
  }

  delete[] result_vec;
  delete[] data1_vec;
  delete[] data2_vec;

  return jetdl::Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}

jetdl::Tensor _math_ops_a_scalar(const jetdl::Tensor& a, const jetdl::Tensor& b,
                                 jetdl::utils::ArithType arith_type) {
  const std::vector<size_t>& shape = b.shape;

  auto result_data = std::make_shared<std::vector<float>>(b.size);

  void (*kernel)(const float*, const float*, float*, const size_t);
  if (arith_type == jetdl::utils::ArithType::ADD) kernel = c_add_a_scalar_cpu;
  if (arith_type == jetdl::utils::ArithType::SUB) kernel = c_sub_a_scalar_cpu;
  if (arith_type == jetdl::utils::ArithType::MUL) kernel = c_mul_a_scalar_cpu;
  if (arith_type == jetdl::utils::ArithType::DIV) kernel = c_div_a_scalar_cpu;

  kernel((*a._data).data(), (*b._data).data(), (*result_data).data(), b.size);

  return jetdl::Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}

jetdl::Tensor _math_ops_b_scalar(const jetdl::Tensor& a, const jetdl::Tensor& b,
                                 jetdl::utils::ArithType arith_type) {
  std::vector<size_t> shape = a.shape;

  auto result_data = std::make_shared<std::vector<float>>(a.size);

  void (*kernel)(const float*, const float*, float*, const size_t);
  if (arith_type == jetdl::utils::ArithType::ADD) kernel = c_add_b_scalar_cpu;
  if (arith_type == jetdl::utils::ArithType::SUB) kernel = c_sub_b_scalar_cpu;
  if (arith_type == jetdl::utils::ArithType::MUL) kernel = c_mul_b_scalar_cpu;
  if (arith_type == jetdl::utils::ArithType::DIV) kernel = c_div_b_scalar_cpu;

  kernel((*a._data).data(), (*b._data).data(), (*result_data).data(), b.size);

  return jetdl::Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}

jetdl::Tensor _math_ops_scalars(const jetdl::Tensor& a, const jetdl::Tensor& b,
                                jetdl::utils::ArithType arith_type) {
  auto result_data = std::make_shared<std::vector<float>>(1);

  void (*kernel)(const float*, const float*, float*);
  if (arith_type == jetdl::utils::ArithType::ADD) kernel = c_add_scalars_cpu;
  if (arith_type == jetdl::utils::ArithType::SUB) kernel = c_sub_scalars_cpu;
  if (arith_type == jetdl::utils::ArithType::MUL) kernel = c_mul_scalars_cpu;
  if (arith_type == jetdl::utils::ArithType::DIV) kernel = c_div_scalars_cpu;

  kernel((*a._data).data(), (*b._data).data(), (*result_data).data());

  return jetdl::Tensor(result_data, {}, a.requires_grad || b.requires_grad);
}
