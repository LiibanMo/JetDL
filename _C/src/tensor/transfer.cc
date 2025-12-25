#include "jetdl/tensor.h"

#include <memory>
#include <stdexcept>

#include "jetdl/device.h"

#ifdef JETDL_WITH_CUDA
#include <cuda_runtime.h>

#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

std::shared_ptr<Tensor> Tensor::to(const Device& target) const {
  // If already on target device, return a shallow copy (shared data)
  if (device == target) {
    auto result = std::make_shared<Tensor>();
    result->_data = _data;
    result->_cuda_data = _cuda_data;
    result->device = device;
    result->ndim = ndim;
    result->shape = shape;
    result->size = size;
    result->strides = strides;
    result->is_contiguous = is_contiguous;
    result->requires_grad = requires_grad;
    // Don't copy grad_fn - this is a new tensor in the graph
    return result;
  }

  // Create new tensor on target device
  auto result = std::make_shared<Tensor>();
  result->ndim = ndim;
  result->shape = shape;
  result->size = size;
  result->strides = strides;
  result->is_contiguous = is_contiguous;
  result->requires_grad = requires_grad;
  result->device = target;

  if (target.is_cuda()) {
    // Transferring to CUDA
#ifdef JETDL_WITH_CUDA
    if (!cuda_is_available()) {
      throw std::runtime_error(
          "CUDA is not available. Cannot transfer tensor to CUDA device.");
    }

    // Allocate GPU memory
    result->_cuda_data = cuda::CUDAAllocator::allocate(size);
    result->_data = nullptr;

    if (device.is_cpu()) {
      // CPU -> CUDA
      cudaMemcpy(result->_cuda_data, _data.get(), size * sizeof(float),
                 cudaMemcpyHostToDevice);
    } else {
      // CUDA -> CUDA (different device, or same device copy)
      cudaMemcpy(result->_cuda_data, _cuda_data, size * sizeof(float),
                 cudaMemcpyDeviceToDevice);
    }
#else
    throw std::runtime_error(
        "JetDL was compiled without CUDA support. "
        "Cannot transfer tensor to CUDA device.");
#endif
  } else {
    // Transferring to CPU
    result->_data = std::shared_ptr<float[]>(new float[size]);
    result->_cuda_data = nullptr;

    if (device.is_cuda()) {
      // CUDA -> CPU
#ifdef JETDL_WITH_CUDA
      cudaMemcpy(result->_data.get(), _cuda_data, size * sizeof(float),
                 cudaMemcpyDeviceToHost);
#else
      throw std::runtime_error(
          "JetDL was compiled without CUDA support. "
          "Cannot transfer from CUDA device.");
#endif
    } else {
      // CPU -> CPU (shouldn't happen due to early return, but handle it)
      std::copy(_data.get(), _data.get() + size, result->_data.get());
    }
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::cuda(int device_id) const {
  return to(Device::cuda(device_id));
}

std::shared_ptr<Tensor> Tensor::cpu() const { return to(Device::cpu()); }

}  // namespace jetdl
