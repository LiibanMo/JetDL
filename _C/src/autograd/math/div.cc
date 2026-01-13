#include <cstddef>
#include <memory>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/autograd/math.h"
#include "jetdl/math.h"
#include "jetdl/math/kernel.h"
#include "jetdl/tensor.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

namespace {

// Fused backward for gradB: result = -a * grad / (b * b)
// Eliminates 4 intermediate tensor allocations
std::shared_ptr<Tensor> fused_grad_B(std::shared_ptr<Tensor>& tensorA,
                                     std::shared_ptr<Tensor>& tensorB,
                                     std::shared_ptr<Tensor>& grad_tensor) {
  const size_t N = grad_tensor->size;
  const bool on_cuda = grad_tensor->device.is_cuda();

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(N);
    c_div_backward_b_cuda(tensorA->_cuda_data, tensorB->_cuda_data,
                          grad_tensor->_cuda_data, result_cuda, N);
    result_tensor = std::make_shared<Tensor>(result_cuda, grad_tensor->shape,
                                             false, grad_tensor->device);
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[N]());
    c_div_backward_b_cpu(tensorA->_data.get(), tensorB->_data.get(),
                         grad_tensor->_data.get(), result_data.get(), N);
    result_tensor =
        std::make_shared<Tensor>(result_data, grad_tensor->shape, false);
  }

  return result_tensor;
}

}  // namespace

DivBackward::DivBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
                         std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions =
      std::vector<std::shared_ptr<Function>>{a->grad_fn, b->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{a, b};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> DivBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& tensorA = this->saved_tensors[0];
  std::shared_ptr<Tensor>& tensorB = this->saved_tensors[1];

  auto grads = std::vector<std::shared_ptr<Tensor>>(2, nullptr);

  // Use fused kernel when shapes match (no broadcasting)
  const bool can_fuse = (grad_tensor->shape == tensorA->shape &&
                         grad_tensor->shape == tensorB->shape);

  if (tensorA->requires_grad) {
    std::shared_ptr<Tensor> gradA = math::div(grad_tensor, tensorB);
    grads[0] = math::sum_to_shape(gradA, tensorA->shape);
  }
  if (tensorB->requires_grad) {
    if (can_fuse) {
      // Fused path: single kernel call instead of 4 operations
      std::shared_ptr<Tensor> gradB = fused_grad_B(tensorA, tensorB, grad_tensor);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    } else {
      // Fallback for broadcasting cases
      std::shared_ptr<Tensor> numer = math::mul(tensorA, grad_tensor);
      numer = math::neg(numer);
      std::shared_ptr<Tensor> denom = math::mul(tensorB, tensorB);
      std::shared_ptr<Tensor> gradB = math::div(numer, denom);
      grads[1] = math::sum_to_shape(gradB, tensorB->shape);
    }
  }

  return grads;
}

}  // namespace jetdl
