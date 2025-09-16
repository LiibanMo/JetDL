#include <cstring>
#include <vector>

#include "jetdl/linalg/kernel.h"
#include "jetdl/linalg/product.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

jetdl::Tensor _linalg_dot(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  auto result_data = std::make_shared<std::vector<float>>(1, 0.0f);
  for (size_t i = 0; i < a.size; i++) {
    result_data->at(0) += a._data->at(i) * b._data->at(i);
  }
  return jetdl::Tensor(result_data, {}, a.requires_grad || b.requires_grad);
}

jetdl::Tensor _linalg_matvec(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  const std::vector<size_t>& shape = jetdl::utils::get_result_shape(
      a.shape, b.shape, jetdl::utils::OpType::MATMUL);

  const size_t M = a.shape[a.ndim - 2];
  const size_t N = b.shape[0];

  const size_t DATA1_ROWS = jetdl::utils::get_next_multiple(M, BLOCK_N_ROWS);
  const size_t BATCH_SIZE = jetdl::utils::get_batch_size(a.shape);

  const size_t DATA1_MAT_SIZE = DATA1_ROWS * N;
  const size_t DATA2_MAT_SIZE = N * BLOCK_N_COLS;
  const size_t RESULT_MAT_SIZE = DATA1_ROWS * BLOCK_N_COLS;

  float* result_matrix = new float[RESULT_MAT_SIZE]();
  float* data1_matrix = new float[DATA1_MAT_SIZE]();
  float* data2_matrix = new float[DATA2_MAT_SIZE]();

  const size_t result_size = a.size / b.size;
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  for (size_t i = 0; i < N; i++) {
    data2_matrix[i * BLOCK_N_COLS] = b._data->at(i);
  }
  for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
    const size_t a_batch_stride = (a.ndim > 2) ? a.strides[a.ndim - 3] : 0;
    const size_t idxA = batch * a_batch_stride;
    std::copy(a._data->begin() + idxA, a._data->begin() + idxA + M * N,
              data1_matrix);
    for (size_t x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
      c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, 0,
                   BLOCK_N_COLS, N);
    }
    for (size_t i = 0; i < M; i++) {
      result_data->at(batch * M + i) = result_matrix[i * BLOCK_N_COLS];
    }
  }

  delete[] result_matrix;
  delete[] data1_matrix;
  delete[] data2_matrix;

  return jetdl::Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}

jetdl::Tensor _linalg_vecmat(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  const std::vector<size_t>& shape = jetdl::utils::get_result_shape(
      a.shape, b.shape, jetdl::utils::OpType::MATMUL);

  const size_t N = a.shape[0];
  const size_t P = b.shape[b.ndim - 1];

  const size_t DATA2_COLS = jetdl::utils::get_next_multiple(P, BLOCK_N_COLS);
  const size_t BATCH_SIZE = jetdl::utils::get_batch_size(b.shape);

  const size_t DATA1_MAT_SIZE = BLOCK_N_ROWS * N;
  const size_t DATA2_MAT_SIZE = N * DATA2_COLS;
  const size_t RESULT_MAT_SIZE = BLOCK_N_ROWS * DATA2_COLS;

  float* result_matrix = new float[RESULT_MAT_SIZE]();
  float* data1_matrix = new float[DATA1_MAT_SIZE]();
  float* data2_matrix = new float[DATA2_MAT_SIZE]();

  const size_t RESULT_SIZE = b.size / a.size;
  auto result_data = std::make_shared<std::vector<float>>(RESULT_SIZE);

  std::copy(a._data->begin(), a._data->end(), data1_matrix);
  for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
    for (size_t i = 0; i < N; i++) {
      const size_t b_batch_stride = (b.ndim > 2) ? b.strides[b.ndim - 3] : 0;
      const size_t idxB = batch * b_batch_stride + i * b.strides[b.ndim - 2];
      std::copy(b._data->begin() + idxB, b._data->begin() + idxB + P,
                data2_matrix + i * DATA2_COLS);
    }
    for (size_t y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
      c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, 0, y, DATA2_COLS,
                   N);
    }
    std::copy(result_matrix, result_matrix + P,
              result_data->begin() + batch * P);
  }

  delete[] result_matrix;
  delete[] data1_matrix;
  delete[] data2_matrix;

  return jetdl::Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}

jetdl::Tensor _linalg_matmul(const jetdl::Tensor& a, const jetdl::Tensor& b) {
  const std::vector<size_t>& shape = jetdl::utils::get_result_shape(
      a.shape, b.shape, jetdl::utils::OpType::MATMUL);

  const size_t M = a.shape[a.ndim - 2];
  const size_t N = a.shape[a.ndim - 1];
  const size_t P = b.shape[b.ndim - 1];

  const size_t BATCH_SIZE = jetdl::utils::get_batch_size(shape);

  const size_t DATA1_ROWS = jetdl::utils::get_next_multiple(M, BLOCK_N_ROWS);
  const size_t DATA2_COLS = jetdl::utils::get_next_multiple(P, BLOCK_N_COLS);

  const size_t RESULT_MAT_SIZE = DATA1_ROWS * DATA2_COLS;
  const size_t DATA1_MAT_SIZE = DATA1_ROWS * N;
  const size_t DATA2_MAT_SIZE = N * DATA2_COLS;

  const auto& strides_pair =
      jetdl::utils::get_strides(a.shape, b.shape, jetdl::utils::OpType::MATMUL);
  const std::vector<size_t>& stridesA = strides_pair.first;
  const std::vector<size_t>& stridesB = strides_pair.second;

  const std::vector<size_t>& idxsA = jetdl::utils::populate_linear_idxs(
      shape, stridesA, jetdl::utils::OpType::MATMUL);
  const std::vector<size_t>& idxsB = jetdl::utils::populate_linear_idxs(
      shape, stridesB, jetdl::utils::OpType::MATMUL);

  float* result_matrix = new float[RESULT_MAT_SIZE]();
  float* data1_matrix = new float[DATA1_MAT_SIZE]();
  float* data2_matrix = new float[DATA2_MAT_SIZE]();

  const size_t RESULT_SIZE = jetdl::utils::get_size(shape);
  auto result_data = std::make_shared<std::vector<float>>(RESULT_SIZE);

  for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
    const size_t idxA = idxsA[batch];
    std::copy(a._data->begin() + idxA, a._data->begin() + idxA + M * N,
              data1_matrix);
    for (size_t i = 0; i < N; i++) {
      const size_t idxB = idxsB[batch] + i * b.strides[b.ndim - 2];
      std::copy(b._data->begin() + idxB, b._data->begin() + idxB + P,
                data2_matrix + i * DATA2_COLS);
    }
    for (size_t x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
      for (size_t y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
        c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, y,
                     DATA2_COLS, N);
      }
    }
    for (size_t i = 0; i < M; i++) {
      const size_t idx = batch * M * P + i * P;
      std::copy(result_matrix + i * DATA2_COLS,
                result_matrix + i * DATA2_COLS + P, result_data->begin() + idx);
    }
  }

  delete[] result_matrix;
  delete[] data1_matrix;
  delete[] data2_matrix;

  return jetdl::Tensor(result_data, shape, a.requires_grad || b.requires_grad);
}
