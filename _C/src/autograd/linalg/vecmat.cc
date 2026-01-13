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

// Fused matvec for grad_A: dA = B @ grad
// B is (M, N), grad is (N,), result is (M,)
// Eliminates mT intermediate by using matvec directly
// Note: grad @ B^T = B @ grad for this dimension arrangement
std::shared_ptr<Tensor> fused_grad_A(std::shared_ptr<Tensor>& tensorB,
                                     std::shared_ptr<Tensor>& grad_tensor) {
  const size_t M = tensorB->shape[0];
  const size_t N = tensorB->shape[1];
  const std::vector<size_t> result_shape = {M};
  const size_t result_size = M;

  const bool on_cuda = grad_tensor->device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
    matvec_kernel_cuda(tensorB->_cuda_data, grad_tensor->_cuda_data,
                       result_cuda, M, N, N);
    result_tensor = std::make_shared<Tensor>(result_cuda, result_shape, false,
                                             grad_tensor->device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    matvec_kernel_cpu(tensorB->_data.get(), grad_tensor->_data.get(),
                      result_data.get(), M, N, N);
    result_tensor = std::make_shared<Tensor>(result_data, result_shape, false);
  }

  return result_tensor;
}

// Fused outer product for grad_B: dB = x ⊗ grad
// x is (M,), grad is (N,), result is (M, N)
// Eliminates unsqueeze intermediates and uses BLAS sger directly
std::shared_ptr<Tensor> fused_grad_B(std::shared_ptr<Tensor>& tensorA,
                                     std::shared_ptr<Tensor>& grad_tensor) {
  const size_t M = tensorA->shape[0];
  const size_t N = grad_tensor->shape[0];
  const std::vector<size_t> result_shape = {M, N};
  const size_t result_size = M * N;

  const bool on_cuda = grad_tensor->device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
    outer_product_kernel_cuda(tensorA->_cuda_data, grad_tensor->_cuda_data,
                              result_cuda, M, N);
    result_tensor = std::make_shared<Tensor>(result_cuda, result_shape, false,
                                             grad_tensor->device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    outer_product_kernel_cpu(tensorA->_data.get(), grad_tensor->_data.get(),
                             result_data.get(), M, N);
    result_tensor = std::make_shared<Tensor>(result_data, result_shape, false);
  }

  return result_tensor;
}

}  // namespace

VecMatBackward::VecMatBackward(std::shared_ptr<Tensor>& a,
                               std::shared_ptr<Tensor>& b,
                               std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> VecMatBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);

  // Use fused kernels for 1D vector @ 2D matrix case
  const bool can_fuse = (tensorA->ndim == 1 && tensorB->ndim == 2 &&
                         grad_tensor->ndim == 1);

  if (tensorA->requires_grad) {
    if (can_fuse) {
      // Fused matvec: dA = B @ grad (equivalent to grad @ B^T)
      std::shared_ptr<Tensor> gradA = fused_grad_A(tensorB, grad_tensor);
      grads[0] = math::sum_to_shape(gradA, tensorA->shape);
    } else {
      // Fallback for batched case
      std::shared_ptr<Tensor> tensorB_mT = linalg::mT(tensorB);
      std::shared_ptr<Tensor> gradA = linalg::matmul(grad_tensor, tensorB_mT);
      grads[0] = math::sum_to_shape(gradA, tensorA->shape);
    }
  }

  if (tensorB->requires_grad) {
    if (can_fuse) {
      // Fused outer product: dB = x ⊗ grad
      std::shared_ptr<Tensor> gradB = fused_grad_B(tensorA, grad_tensor);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    } else {
      // Fallback for batched case
      std::shared_ptr<Tensor> unsqueezed_tensorA = unsqueeze(tensorA, -1);
      std::shared_ptr<Tensor> unsqueezed_grad_tensor = unsqueeze(grad_tensor, 0);
      std::shared_ptr<Tensor> gradB =
          linalg::matmul(unsqueezed_tensorA, unsqueezed_grad_tensor);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    }
  }

  return grads;
}

}  // namespace jetdl
