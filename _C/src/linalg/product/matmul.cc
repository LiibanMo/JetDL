#include <cstring>
#include <memory>
#include <vector>

#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg/kernel.h"
#include "jetdl/linalg/product.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> _linalg_dot(std::shared_ptr<Tensor>& a,
                                    std::shared_ptr<Tensor>& b) {
  auto result_data = std::make_shared<std::vector<float>>(1, 0.0f);

  for (size_t i = 0; i < a->size; i++) {
    result_data->at(0) += a->_data->at(i) * b->_data->at(i);
  }

  auto result_tensor = std::make_shared<Tensor>(
      result_data, std::vector<size_t>{}, a->requires_grad || b->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn = std::make_shared<DotBackward>(a, b, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _linalg_matvec(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b) {
  const std::vector<size_t>& shape =
      utils::get_result_shape(a->shape, b->shape, utils::OpType::MATMUL);

  const size_t M = a->shape[a->ndim - 2];
  const size_t N = b->shape[0];

  const size_t data1_rows = utils::get_next_multiple(M, BLOCK_N_ROWS);
  const size_t batch_size = utils::get_batch_size(a->shape);

  const size_t data1_mat_size = data1_rows * N;
  const size_t data2_mat_size = N * BLOCK_N_COLS;
  const size_t result_mat_size = data1_rows * BLOCK_N_COLS;

  float* result_matrix = new float[result_mat_size]();
  float* data1_matrix = new float[data1_mat_size]();
  float* data2_matrix = new float[data2_mat_size]();

  const size_t result_size = a->size / b->size;
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  for (size_t i = 0; i < N; i++) {
    data2_matrix[i * BLOCK_N_COLS] = b->_data->at(i);
  }
  for (size_t batch = 0; batch < batch_size; batch++) {
    const size_t a_batch_stride = (a->ndim > 2) ? a->strides[a->ndim - 3] : 0;
    const size_t idxA = batch * a_batch_stride;
    std::copy(a->_data->begin() + idxA, a->_data->begin() + idxA + M * N,
              data1_matrix);
    for (size_t x = 0; x < data1_rows; x += BLOCK_N_ROWS) {
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

  auto result_tensor = std::make_shared<Tensor>(
      result_data, shape, a->requires_grad || b->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<MatVecBackward>(a, b, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _linalg_vecmat(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b) {
  const std::vector<size_t>& shape =
      utils::get_result_shape(a->shape, b->shape, utils::OpType::MATMUL);

  const size_t N = a->shape[0];
  const size_t P = b->shape[b->ndim - 1];

  const size_t data2_cols = utils::get_next_multiple(P, BLOCK_N_COLS);
  const size_t batch_size = utils::get_batch_size(b->shape);

  const size_t data1_mat_size = BLOCK_N_ROWS * N;
  const size_t data2_mat_size = N * data2_cols;
  const size_t result_mat_size = BLOCK_N_ROWS * data2_cols;

  float* result_matrix = new float[result_mat_size]();
  float* data1_matrix = new float[data1_mat_size]();
  float* data2_matrix = new float[data2_mat_size]();

  const size_t result_size = b->size / a->size;
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  std::copy(a->_data->begin(), a->_data->end(), data1_matrix);
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (size_t i = 0; i < N; i++) {
      const size_t b_batch_stride = (b->ndim > 2) ? b->strides[b->ndim - 3] : 0;
      const size_t idxB = batch * b_batch_stride + i * b->strides[b->ndim - 2];
      std::copy(b->_data->begin() + idxB, b->_data->begin() + idxB + P,
                data2_matrix + i * data2_cols);
    }
    for (size_t y = 0; y < data2_cols; y += BLOCK_N_COLS) {
      c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, 0, y, data2_cols,
                   N);
    }
    std::copy(result_matrix, result_matrix + P,
              result_data->begin() + batch * P);
  }

  delete[] result_matrix;
  delete[] data1_matrix;
  delete[] data2_matrix;

  auto result_tensor = std::make_shared<Tensor>(
      result_data, shape, a->requires_grad || b->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<VecMatBackward>(a, b, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _linalg_matmul(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b) {
  const std::vector<size_t>& shape =
      utils::get_result_shape(a->shape, b->shape, utils::OpType::MATMUL);

  const size_t M = a->shape[a->ndim - 2];
  const size_t N = a->shape[a->ndim - 1];
  const size_t P = b->shape[b->ndim - 1];

  const size_t batch_size = utils::get_batch_size(shape);

  const size_t data1_rows = utils::get_next_multiple(M, BLOCK_N_ROWS);
  const size_t data2_cols = utils::get_next_multiple(P, BLOCK_N_COLS);

  const size_t result_mat_size = data1_rows * data2_cols;
  const size_t data1_mat_size = data1_rows * N;
  const size_t data2_mat_size = N * data2_cols;

  const auto& strides_pair =
      utils::get_strides(a->shape, b->shape, utils::OpType::MATMUL);
  const std::vector<size_t>& stridesA = strides_pair.first;
  const std::vector<size_t>& stridesB = strides_pair.second;

  const std::vector<size_t>& idxsA =
      utils::populate_linear_idxs(shape, stridesA, utils::OpType::MATMUL);
  const std::vector<size_t>& idxsB =
      utils::populate_linear_idxs(shape, stridesB, utils::OpType::MATMUL);

  float* result_matrix = new float[result_mat_size]();
  float* data1_matrix = new float[data1_mat_size]();
  float* data2_matrix = new float[data2_mat_size]();

  const size_t result_size = utils::get_size(shape);
  auto result_data = std::make_shared<std::vector<float>>(result_size);

  for (size_t batch = 0; batch < batch_size; batch++) {
    const size_t idxA = idxsA[batch];
    std::copy(a->_data->begin() + idxA, a->_data->begin() + idxA + M * N,
              data1_matrix);
    for (size_t i = 0; i < N; i++) {
      const size_t idxB = idxsB[batch] + i * b->strides[b->ndim - 2];
      std::copy(b->_data->begin() + idxB, b->_data->begin() + idxB + P,
                data2_matrix + i * data2_cols);
    }
    for (size_t x = 0; x < data1_rows; x += BLOCK_N_ROWS) {
      for (size_t y = 0; y < data2_cols; y += BLOCK_N_COLS) {
        c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, y,
                     data2_cols, N);
      }
    }
    for (size_t i = 0; i < M; i++) {
      const size_t idx = batch * M * P + i * P;
      std::copy(result_matrix + i * data2_cols,
                result_matrix + i * data2_cols + P, result_data->begin() + idx);
    }
  }

  delete[] result_matrix;
  delete[] data1_matrix;
  delete[] data2_matrix;

  auto result_tensor = std::make_shared<Tensor>(
      result_data, shape, a->requires_grad || b->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<MatmulBackward>(a, b, result_tensor);
  }

  return result_tensor;
}

}  // namespace jetdl
