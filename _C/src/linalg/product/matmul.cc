#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef JETDL_WITH_OPENMP
#include <omp.h>
#endif

#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg/kernel.h"
#include "jetdl/linalg/product.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

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

  const Device& device = a->device;
  const bool requires_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> result_tensor;

  if (device.is_cuda()) {
#ifdef JETDL_WITH_CUDA
    // For CUDA, reshape the (1,1) result to scalar shape
    // Reuse the existing CUDA data by creating a new tensor with scalar shape
    result_tensor = std::make_shared<Tensor>();
    result_tensor->_cuda_data = view_result_tensor->_cuda_data;
    result_tensor->_data = nullptr;
    result_tensor->device = device;
    result_tensor->ndim = 0;
    result_tensor->shape = {};
    result_tensor->size = 1;
    result_tensor->strides = {};
    result_tensor->is_contiguous = true;
    result_tensor->requires_grad = requires_grad;
    // Prevent double-free by nulling out source's cuda_data
    view_result_tensor->_cuda_data = nullptr;
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    result_tensor = std::make_shared<Tensor>(
        view_result_tensor->_data[0], requires_grad);
  }

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
  // Device validation
  if (tensor1->device != tensor2->device) {
    throw std::runtime_error(
        "Cannot perform matmul between tensors on different devices. "
        "Tensor 1 is on " + tensor1->device.str() + ", tensor 2 is on " +
        tensor2->device.str() + ". Use .to(), .cuda(), or .cpu() to move tensors.");
  }

  const Device& device = tensor1->device;
  const bool on_cuda = device.is_cuda();
  const bool requires_grad = tensor1->requires_grad || tensor2->requires_grad;

  // Fast path for 2D matmul
  if (tensor1->ndim == 2 && tensor2->ndim == 2) {
    const size_t M = tensor1->shape[0];
    const size_t K = tensor1->shape[1];
    const size_t N = tensor2->shape[1];

    const std::vector<size_t> shape = {M, N};
    const size_t result_size = M * N;

    std::shared_ptr<Tensor> result_tensor;

    if (on_cuda) {
#ifdef JETDL_WITH_CUDA
      float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
      matmul_kernel_cuda(tensor1->get(), tensor2->get(), result_cuda,
                         M, K, N, K, N, N);
      result_tensor = std::make_shared<Tensor>(result_cuda, shape, requires_grad, device);
#else
      throw std::runtime_error("JetDL compiled without CUDA support");
#endif
    } else {
      auto result_data = std::shared_ptr<float[]>(new float[result_size]());
      matmul_kernel_cpu(tensor1->_data.get(), tensor2->_data.get(), result_data.get(),
                        M, K, N, K, N, N);
      result_tensor = std::make_shared<Tensor>(result_data, shape, requires_grad);
    }

    if (result_tensor->requires_grad) {
      result_tensor->grad_fn =
          std::make_shared<MatmulBackward>(tensor1, tensor2, result_tensor);
    }
    return result_tensor;
  }

  // Batched matmul path
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

  const std::vector<size_t>& idxs1 =
      utils::populate_linear_idxs(shape, batch_strides1, utils::OpType::MATMUL);
  const std::vector<size_t>& idxs2 =
      utils::populate_linear_idxs(shape, batch_strides2, utils::OpType::MATMUL);

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
    float* ptr1 = tensor1->get();
    float* ptr2 = tensor2->get();

    // CUDA path: process batches sequentially (GPU handles parallelism)
    for (size_t b = 0; b < B; b++) {
      float* ptr1_b = ptr1 + idxs1[b];
      float* ptr2_b = ptr2 + idxs2[b];
      float* result_ptr_b = result_cuda + b * M * N;
      matmul_kernel_cuda(ptr1_b, ptr2_b, result_ptr_b, M, K, N, K, N, N);
    }

    result_tensor = std::make_shared<Tensor>(result_cuda, shape, requires_grad, device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    // CPU path: parallel with OpenMP
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    float* ptr1 = tensor1->_data.get();
    float* ptr2 = tensor2->_data.get();
    float* result_ptr = result_data.get();

#ifdef JETDL_WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t b = 0; b < B; b++) {
      float* ptr1_b = ptr1 + idxs1[b];
      float* ptr2_b = ptr2 + idxs2[b];
      float* result_ptr_b = result_ptr + b * M * N;
      matmul_kernel_cpu(ptr1_b, ptr2_b, result_ptr_b, M, K, N, K, N, N);
    }

    result_tensor = std::make_shared<Tensor>(result_data, shape, requires_grad);
  }

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn =
        std::make_shared<MatmulBackward>(tensor1, tensor2, result_tensor);
  }

  return result_tensor;
}
}  // namespace jetdl
