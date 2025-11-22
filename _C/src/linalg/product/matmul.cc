#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg/kernel.h"
#include "jetdl/linalg/product.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

namespace jetdl {

std::shared_ptr<Tensor> _linalg_dot(std::shared_ptr<Tensor>& a,
                                    std::shared_ptr<Tensor>& b) {
  std::vector<size_t> view_shape_a = {1};
  view_shape_a.push_back(a->shape[0]);

  std::vector<size_t> view_shape_b = b->shape;
  view_shape_b.push_back(1);

  std::shared_ptr<Tensor> view_tensor_a = view(a, view_shape_a);
  std::shared_ptr<Tensor> view_tensor_b = view(b, view_shape_b);

  std::shared_ptr<Tensor> view_result_tensor =
      _linalg_matmul(view_tensor_a, view_tensor_b);

  auto result_tensor = std::make_shared<Tensor>(
      view_result_tensor->_data[0], a->requires_grad || b->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn = std::make_shared<DotBackward>(a, b, result_tensor);
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _linalg_matvec(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b) {
  std::vector<size_t> view_shape_b = b->shape;
  view_shape_b.push_back(1);

  std::shared_ptr<Tensor> view_tensor_b = view(b, view_shape_b);

  std::shared_ptr<Tensor> view_result_tensor = _linalg_matmul(a, view_tensor_b);

  const std::vector<size_t>& shape =
      utils::get_result_shape(a->shape, b->shape, utils::OpType::MATMUL);

  std::shared_ptr<Tensor> result_tensor = view(view_result_tensor, shape);
  result_tensor->requires_grad = a->requires_grad || b->requires_grad;

  return result_tensor;
}

std::shared_ptr<Tensor> _linalg_vecmat(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b) {
  std::vector<size_t> view_shape_a = {1};
  view_shape_a.push_back(a->shape[0]);

  std::shared_ptr<Tensor> view_tensor_a = view(a, view_shape_a);

  std::shared_ptr<Tensor> view_result_tensor = _linalg_matmul(view_tensor_a, b);

  const std::vector<size_t>& shape =
      utils::get_result_shape(a->shape, b->shape, utils::OpType::MATMUL);

  std::shared_ptr<Tensor> result_tensor = view(view_result_tensor, shape);
  result_tensor->requires_grad = a->requires_grad || b->requires_grad;

  return result_tensor;
}

std::shared_ptr<Tensor> _linalg_matmul(std::shared_ptr<Tensor>& tensor1,
                                       std::shared_ptr<Tensor>& tensor2) {
  // Fast path for 2D matmul
  if (tensor1->ndim == 2 && tensor2->ndim == 2) {
    const size_t M = tensor1->shape[0];
    const size_t K = tensor1->shape[1];
    const size_t N = tensor2->shape[1];

    const std::vector<size_t> shape = {M, N};
    const size_t result_size = M * N;
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());

    matmul_kernel(tensor1->_data.get(), tensor2->_data.get(), result_data.get(),
                  M, K, N, K, N, N);

    auto result_tensor = std::make_shared<Tensor>(
        result_data, shape, tensor1->requires_grad || tensor2->requires_grad);

    if (result_tensor->requires_grad) {
      result_tensor->grad_fn =
          std::make_shared<MatmulBackward>(tensor1, tensor2, result_tensor);
    }
    return result_tensor;
  }

  const std::vector<size_t>& shape = utils::get_result_shape(
      tensor1->shape, tensor2->shape, utils::OpType::MATMUL);

  const size_t M = tensor1->shape[tensor1->ndim - 2];
  const size_t K = tensor1->shape[tensor1->ndim - 1];
  const size_t N = tensor2->shape[tensor2->ndim - 1];

  const size_t B = utils::get_batch_size(shape);

  const auto& strides_pair =
      utils::get_strides(tensor1->shape, tensor2->shape, utils::OpType::MATMUL);
  const std::vector<size_t>& batch_strides1 = strides_pair.first;
  const std::vector<size_t>& batch_strides2 = strides_pair.second;

  const size_t result_size = utils::get_size(shape);
  auto result_data = std::shared_ptr<float[]>(new float[result_size]());

  const std::vector<size_t>& idxs1 =
      utils::populate_linear_idxs(shape, batch_strides1, utils::OpType::MATMUL);
  const std::vector<size_t>& idxs2 =
      utils::populate_linear_idxs(shape, batch_strides2, utils::OpType::MATMUL);

  float* ptr1 = tensor1->_data.get();
  float* ptr2 = tensor2->_data.get();
  float* result_ptr = result_data.get();

  auto worker = [&](size_t start_b, size_t end_b) {
    for (size_t b = start_b; b < end_b; b++) {
      float* ptr1_b = ptr1 + idxs1[b];
      float* ptr2_b = ptr2 + idxs2[b];
      float* result_ptr_b = result_ptr + b * M * N;
      matmul_kernel(ptr1_b, ptr2_b, result_ptr_b, M, K, N, K, N, N);
    }
  };

  if (B > 0) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (B < num_threads) {
      num_threads = B;
    }

    if (num_threads > 1) {
      std::vector<std::thread> threads;
      threads.reserve(num_threads);
      size_t chunk_size = (B + num_threads - 1) / num_threads;
      for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start_b = i * chunk_size;
        size_t end_b = std::min(start_b + chunk_size, B);
        if (start_b < end_b) {
          threads.emplace_back(worker, start_b, end_b);
        }
      }
      for (auto& t : threads) {
        t.join();
      }
    } else {
      worker(0, B);
    }
  }

  auto result_tensor = std::make_shared<Tensor>(
      result_data, shape, tensor1->requires_grad || tensor2->requires_grad);

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<MatmulBackward>(tensor1, tensor2, result_tensor);
  }

  return result_tensor;
}
}  // namespace jetdl
