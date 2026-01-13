#include <cstddef>
#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg.h"
#include "jetdl/linalg/kernel.h"
#include "jetdl/math.h"
#include "jetdl/tensor.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

namespace {

// Fused backward for dA = dC @ B^T
// dC is (M, N), B is (K, N), result dA is (M, K)
// Uses matmul_nt kernel to avoid creating B^T intermediate
std::shared_ptr<Tensor> fused_grad_A(std::shared_ptr<Tensor>& grad_tensor,
                                     std::shared_ptr<Tensor>& tensorB) {
  // grad_tensor: (M, N), tensorB: (K, N)
  // dA = grad_tensor @ tensorB^T = (M, N) @ (N, K) = (M, K)
  const size_t M = grad_tensor->shape[grad_tensor->ndim - 2];
  const size_t N = grad_tensor->shape[grad_tensor->ndim - 1];
  const size_t K = tensorB->shape[tensorB->ndim - 2];

  const std::vector<size_t> result_shape = {M, K};
  const size_t result_size = M * K;

  const bool on_cuda = grad_tensor->device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
    // matmul_nt: C = A @ B^T where A is (M, K_inner), B is (N, K_inner)
    // Here: dA = dC @ B^T, dC is (M, N), B is (K, N)
    // So K_inner = N, and we want (M, K) result
    // Actually for NT: A is (M, K_inner), B is (result_cols, K_inner), C is (M, result_cols)
    // dC (M, N) @ B^T (N, K) => M=M, K_inner=N, result_cols=K
    matmul_nt_kernel_cuda(grad_tensor->get(), tensorB->get(), result_cuda, M, N,
                          K, N, N, K);
    result_tensor = std::make_shared<Tensor>(result_cuda, result_shape, false,
                                             grad_tensor->device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    matmul_nt_kernel_cpu(grad_tensor->_data.get(), tensorB->_data.get(),
                         result_data.get(), M, N, K, N, N, K);
    result_tensor = std::make_shared<Tensor>(result_data, result_shape, false);
  }

  return result_tensor;
}

// Fused backward for dB = A^T @ dC
// A is (M, K), dC is (M, N), result dB is (K, N)
// Uses matmul_tn kernel to avoid creating A^T intermediate
std::shared_ptr<Tensor> fused_grad_B(std::shared_ptr<Tensor>& tensorA,
                                     std::shared_ptr<Tensor>& grad_tensor) {
  // tensorA: (M, K), grad_tensor: (M, N)
  // dB = tensorA^T @ grad_tensor = (K, M) @ (M, N) = (K, N)
  const size_t M = tensorA->shape[tensorA->ndim - 2];
  const size_t K = tensorA->shape[tensorA->ndim - 1];
  const size_t N = grad_tensor->shape[grad_tensor->ndim - 1];

  const std::vector<size_t> result_shape = {K, N};
  const size_t result_size = K * N;

  const bool on_cuda = tensorA->device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);
    // matmul_tn: C = A^T @ B where A is (K_inner, M_out), B is (K_inner, N)
    // Here: dB = A^T @ dC, A is (M, K), dC is (M, N)
    // So K_inner = M, M_out = K, N = N
    matmul_tn_kernel_cuda(tensorA->get(), grad_tensor->get(), result_cuda, K, M,
                          N, K, N, N);
    result_tensor = std::make_shared<Tensor>(result_cuda, result_shape, false,
                                             tensorA->device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    matmul_tn_kernel_cpu(tensorA->_data.get(), grad_tensor->_data.get(),
                         result_data.get(), K, M, N, K, N, N);
    result_tensor = std::make_shared<Tensor>(result_data, result_shape, false);
  }

  return result_tensor;
}

}  // namespace

MatmulBackward::MatmulBackward(std::shared_ptr<Tensor>& a,
                               std::shared_ptr<Tensor>& b,
                               std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> MatmulBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);

  // Only handle 2D case with fused kernels for now
  // Batched case falls back to the original implementation
  if (grad_tensor->ndim == 2 && tensorA->ndim == 2 && tensorB->ndim == 2) {
    if (tensorA->requires_grad) {
      // dA = dC @ B^T (fused, no intermediate tensor)
      std::shared_ptr<Tensor> gradA = fused_grad_A(grad_tensor, tensorB);
      grads[0] = math::sum_to_shape(gradA, tensorA->shape);
    }
    if (tensorB->requires_grad) {
      // dB = A^T @ dC (fused, no intermediate tensor)
      std::shared_ptr<Tensor> gradB = fused_grad_B(tensorA, grad_tensor);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    }
  } else {
    // Fallback for batched matmul
    if (tensorA->requires_grad) {
      std::shared_ptr<Tensor> tensorB_mT = linalg::mT(tensorB);
      std::shared_ptr<Tensor> gradA = linalg::matmul(grad_tensor, tensorB_mT);
      grads[0] = math::sum_to_shape(gradA, tensorA->shape);
    }
    if (tensorB->requires_grad) {
      std::shared_ptr<Tensor> tensorA_mT = linalg::mT(tensorA);
      std::shared_ptr<Tensor> gradB = linalg::matmul(tensorA_mT, grad_tensor);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    }
  }

  return grads;
}

}  // namespace jetdl
