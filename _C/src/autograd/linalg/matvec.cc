#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg.h"
#include "jetdl/linalg/kernel.h"
#include "jetdl/math.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

namespace {

// Fused outer product for grad_A: dA = grad ⊗ x
// grad is (M,), x is (N,), result is (M, N)
// Eliminates unsqueeze intermediates and uses BLAS sger directly
std::shared_ptr<Tensor> fused_grad_A(std::shared_ptr<Tensor>& grad_tensor,
                                     std::shared_ptr<Tensor>& tensorB) {
  const size_t M = grad_tensor->shape[0];
  const size_t N = tensorB->shape[0];
  const std::vector<size_t> result_shape = {M, N};
  const size_t result_size = M * N;

  const bool on_cuda = grad_tensor->device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
    outer_product_kernel_cuda(grad_tensor->_cuda_data, tensorB->_cuda_data,
                              result_cuda, M, N);
    result_tensor = std::make_shared<Tensor>(result_cuda, result_shape, false,
                                             grad_tensor->device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    outer_product_kernel_cpu(grad_tensor->_data.get(), tensorB->_data.get(),
                             result_data.get(), M, N);
    result_tensor =
        std::make_shared<Tensor>(result_data, result_shape, false);
  }

  return result_tensor;
}

// Fused transposed matvec for grad_B: dB = A^T @ grad
// A is (M, N), grad is (M,), result is (N,)
// Eliminates mT intermediate and uses BLAS sgemv with Trans directly
std::shared_ptr<Tensor> fused_grad_B(std::shared_ptr<Tensor>& tensorA,
                                     std::shared_ptr<Tensor>& grad_tensor) {
  const size_t M = tensorA->shape[0];
  const size_t N = tensorA->shape[1];
  const std::vector<size_t> result_shape = {N};
  const size_t result_size = N;

  const bool on_cuda = tensorA->device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
    matvec_t_kernel_cuda(tensorA->_cuda_data, grad_tensor->_cuda_data,
                         result_cuda, M, N, N);
    result_tensor = std::make_shared<Tensor>(result_cuda, result_shape, false,
                                             tensorA->device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    matvec_t_kernel_cpu(tensorA->_data.get(), grad_tensor->_data.get(),
                        result_data.get(), M, N, N);
    result_tensor = std::make_shared<Tensor>(result_data, result_shape, false);
  }

  return result_tensor;
}

}  // namespace

MatVecBackward::MatVecBackward(std::shared_ptr<Tensor>& a,
                               std::shared_ptr<Tensor>& b,
                               std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> MatVecBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);

  // Use fused kernels for 2D matrix @ 1D vector case
  const bool can_fuse = (tensorA->ndim == 2 && tensorB->ndim == 1 &&
                         grad_tensor->ndim == 1);

  if (tensorA->requires_grad) {
    if (can_fuse) {
      // Fused outer product: dA = grad ⊗ x
      std::shared_ptr<Tensor> gradA = fused_grad_A(grad_tensor, tensorB);
      grads[0] = math::sum_to_shape(gradA, tensorA->shape);
    } else {
      // Fallback for batched case
      std::shared_ptr<Tensor> unsqueezed_tensorB = unsqueeze(tensorB, 0);
      std::shared_ptr<Tensor> unsqueezed_grad_tensor = unsqueeze(grad_tensor, -1);
      std::shared_ptr<Tensor> gradA =
          linalg::matmul(unsqueezed_grad_tensor, unsqueezed_tensorB);
      grads[0] = math::sum_to_shape(gradA, tensorA->shape);
    }
  }

  if (tensorB->requires_grad) {
    if (can_fuse) {
      // Fused transposed matvec: dB = A^T @ grad
      std::shared_ptr<Tensor> gradB = fused_grad_B(tensorA, grad_tensor);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    } else {
      // Fallback for batched case
      std::shared_ptr<Tensor> tensorA_mT = linalg::mT(tensorA);
      std::shared_ptr<Tensor> gradB = linalg::matmul(tensorA_mT, grad_tensor);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    }
  }

  return grads;
}

}  // namespace jetdl
