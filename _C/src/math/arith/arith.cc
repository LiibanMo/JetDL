#include "jetdl/math/arith.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef JETDL_WITH_OPENMP
#include <omp.h>
#endif

#include "jetdl/autograd/math.h"
#include "jetdl/math/kernel.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

using ArithType = utils::ArithType;
using OpType = utils::OpType;

std::shared_ptr<Tensor> _math_ops(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b,
                                  ArithType arith_type) {
  // Device validation
  if (a->device != b->device) {
    throw std::runtime_error(
        "Cannot perform arithmetic between tensors on different devices. "
        "Tensor a is on " + a->device.str() + ", tensor b is on " +
        b->device.str() + ". Use .to(), .cuda(), or .cpu() to move tensors.");
  }

  const Device& device = a->device;
  const bool on_cuda = device.is_cuda();

  const std::vector<size_t>& shape =
      utils::get_result_shape(a->shape, b->shape, OpType::ARITHMETIC);

  const auto& strides_pair =
      utils::get_strides(a->shape, b->shape, OpType::ARITHMETIC);
  const std::vector<size_t>& strides_a = strides_pair.first;
  const std::vector<size_t>& strides_b = strides_pair.second;

  const size_t result_size = utils::get_size(shape);
  const bool requires_grad = a->requires_grad || b->requires_grad;

  // Select kernel functions based on device
  void (*kernel)(const float*, const float*, float*, const size_t);
  void (*kernel_a_scalar)(const float*, const float*, float*, const size_t);
  void (*kernel_b_scalar)(const float*, const float*, float*, const size_t);

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    if (arith_type == ArithType::ADD) {
      kernel = c_add_cuda;
      kernel_a_scalar = c_add_a_scalar_cuda;
      kernel_b_scalar = c_add_b_scalar_cuda;
    } else if (arith_type == ArithType::SUB) {
      kernel = c_sub_cuda;
      kernel_a_scalar = c_sub_a_scalar_cuda;
      kernel_b_scalar = c_sub_b_scalar_cuda;
    } else if (arith_type == ArithType::MUL) {
      kernel = c_mul_cuda;
      kernel_a_scalar = c_mul_a_scalar_cuda;
      kernel_b_scalar = c_mul_b_scalar_cuda;
    } else if (arith_type == ArithType::DIV) {
      kernel = c_div_cuda;
      kernel_a_scalar = c_div_a_scalar_cuda;
      kernel_b_scalar = c_div_b_scalar_cuda;
    } else {
      throw std::runtime_error(
          "INTERNAL (arith.cc::_math_ops) ArithType type invalid");
    }
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    if (arith_type == ArithType::ADD) {
      kernel = c_add_cpu;
      kernel_a_scalar = c_add_a_scalar_cpu;
      kernel_b_scalar = c_add_b_scalar_cpu;
    } else if (arith_type == ArithType::SUB) {
      kernel = c_sub_cpu;
      kernel_a_scalar = c_sub_a_scalar_cpu;
      kernel_b_scalar = c_sub_b_scalar_cpu;
    } else if (arith_type == ArithType::MUL) {
      kernel = c_mul_cpu;
      kernel_a_scalar = c_mul_a_scalar_cpu;
      kernel_b_scalar = c_mul_b_scalar_cpu;
    } else if (arith_type == ArithType::DIV) {
      kernel = c_div_cpu;
      kernel_a_scalar = c_div_a_scalar_cpu;
      kernel_b_scalar = c_div_b_scalar_cpu;
    } else {
      throw std::runtime_error(
          "INTERNAL (arith.cc::_math_ops) ArithType type invalid");
    }
  }

  const size_t ndim = shape.size();
  const size_t N = shape.back();
  const size_t total_num_rows = result_size / N;

  // Allocate result and execute kernels
  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    // Allocate result on CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(result_size);

    // CUDA path: no CPU threading, GPU handles parallelism
    if (a->shape == b->shape && a->is_contiguous && b->is_contiguous) {
      kernel(a->get(), b->get(), result_cuda, result_size);
    } else {
      // Broadcast case: process row by row on GPU
      std::vector<size_t> back_strides;
      if (ndim > 1) {
        back_strides.resize(ndim - 1);
        back_strides.back() = 1;
        for (int i = ndim - 3; i >= 0; --i) {
          back_strides[i] = back_strides[i + 1] * shape[i + 1];
        }
      }

      for (size_t row = 0; row < total_num_rows; row++) {
        size_t idx_a = 0;
        size_t idx_b = 0;
        if (ndim > 1) {
          size_t temp_row = row;
          for (size_t i = 0; i < ndim - 1; i++) {
            const size_t multi_dim_idx = temp_row / back_strides[i];
            idx_a += multi_dim_idx * strides_a[i];
            idx_b += multi_dim_idx * strides_b[i];
            temp_row %= back_strides[i];
          }
        }

        const float* ptr_a = a->get() + idx_a;
        const float* ptr_b = b->get() + idx_b;
        float* ptr_c = result_cuda + row * N;

        bool a_is_broadcast = (strides_a.back() == 0 && N > 1);
        bool b_is_broadcast = (strides_b.back() == 0 && N > 1);

        if (a_is_broadcast) {
          kernel_a_scalar(ptr_a, ptr_b, ptr_c, N);
        } else if (b_is_broadcast) {
          kernel_b_scalar(ptr_a, ptr_b, ptr_c, N);
        } else {
          kernel(ptr_a, ptr_b, ptr_c, N);
        }
      }
    }

    result_tensor = std::make_shared<Tensor>(result_cuda, shape, requires_grad, device);
#endif
  } else {
    // CPU path: allocate and parallelize with OpenMP
    auto result_data = std::shared_ptr<float[]>(new float[result_size]());
    float* result_ptr = result_data.get();

    // Fast path: contiguous tensors with same shape
    if (a->shape == b->shape && a->is_contiguous && b->is_contiguous) {
      kernel(a->_data.get(), b->_data.get(), result_ptr, result_size);
    } else {
      // Broadcast path: row-by-row with parallelization
      std::vector<size_t> back_strides;
      if (ndim > 1) {
        back_strides.resize(ndim - 1);
        back_strides.back() = 1;
        for (int i = ndim - 3; i >= 0; --i) {
          back_strides[i] = back_strides[i + 1] * shape[i + 1];
        }
      }

#ifdef JETDL_WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t row = 0; row < total_num_rows; row++) {
        size_t idx_a = 0;
        size_t idx_b = 0;
        if (ndim > 1) {
          size_t temp_row = row;
          for (size_t i = 0; i < ndim - 1; i++) {
            const size_t multi_dim_idx = temp_row / back_strides[i];
            idx_a += multi_dim_idx * strides_a[i];
            idx_b += multi_dim_idx * strides_b[i];
            temp_row %= back_strides[i];
          }
        }

        const float* ptr_a = a->_data.get() + idx_a;
        const float* ptr_b = b->_data.get() + idx_b;
        float* ptr_c = result_ptr + row * N;

        bool a_is_broadcast = (strides_a.back() == 0 && N > 1);
        bool b_is_broadcast = (strides_b.back() == 0 && N > 1);

        if (a_is_broadcast) {
          kernel_a_scalar(ptr_a, ptr_b, ptr_c, N);
        } else if (b_is_broadcast) {
          kernel_b_scalar(ptr_a, ptr_b, ptr_c, N);
        } else {
          kernel(ptr_a, ptr_b, ptr_c, N);
        }
      }
    }

    result_tensor = std::make_shared<Tensor>(result_data, shape, requires_grad);
  }

  if (result_tensor->requires_grad) {
    if (arith_type == ArithType::ADD) {
      result_tensor->grad_fn =
          std::make_shared<AddBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::SUB) {
      result_tensor->grad_fn =
          std::make_shared<SubBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::MUL) {
      result_tensor->grad_fn =
          std::make_shared<MulBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::DIV) {
      result_tensor->grad_fn =
          std::make_shared<DivBackward>(a, b, result_tensor);
    }
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _math_ops_a_scalar(std::shared_ptr<Tensor>& a,
                                           std::shared_ptr<Tensor>& b,
                                           ArithType arith_type) {
  // Device validation
  if (a->device != b->device) {
    throw std::runtime_error(
        "Cannot perform arithmetic between tensors on different devices. "
        "Tensor a is on " + a->device.str() + ", tensor b is on " +
        b->device.str() + ". Use .to(), .cuda(), or .cpu() to move tensors.");
  }

  const Device& device = a->device;
  const bool on_cuda = device.is_cuda();
  const std::vector<size_t>& shape = b->shape;
  const bool requires_grad = a->requires_grad || b->requires_grad;

  void (*kernel)(const float*, const float*, float*, const size_t);

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    if (arith_type == ArithType::ADD) kernel = c_add_a_scalar_cuda;
    else if (arith_type == ArithType::SUB) kernel = c_sub_a_scalar_cuda;
    else if (arith_type == ArithType::MUL) kernel = c_mul_a_scalar_cuda;
    else if (arith_type == ArithType::DIV) kernel = c_div_a_scalar_cuda;
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    if (arith_type == ArithType::ADD) kernel = c_add_a_scalar_cpu;
    else if (arith_type == ArithType::SUB) kernel = c_sub_a_scalar_cpu;
    else if (arith_type == ArithType::MUL) kernel = c_mul_a_scalar_cpu;
    else if (arith_type == ArithType::DIV) kernel = c_div_a_scalar_cpu;
  }

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(b->size);
    kernel(a->get(), b->get(), result_cuda, b->size);
    result_tensor = std::make_shared<Tensor>(result_cuda, shape, requires_grad, device);
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[b->size]());
    kernel(a->_data.get(), b->_data.get(), result_data.get(), b->size);
    result_tensor = std::make_shared<Tensor>(result_data, shape, requires_grad);
  }

  if (result_tensor->requires_grad) {
    if (arith_type == ArithType::ADD) {
      result_tensor->grad_fn =
          std::make_shared<AddBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::SUB) {
      result_tensor->grad_fn =
          std::make_shared<SubBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::MUL) {
      result_tensor->grad_fn =
          std::make_shared<MulBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::DIV) {
      result_tensor->grad_fn =
          std::make_shared<DivBackward>(a, b, result_tensor);
    }
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _math_ops_b_scalar(std::shared_ptr<Tensor>& a,
                                           std::shared_ptr<Tensor>& b,
                                           ArithType arith_type) {
  // Device validation
  if (a->device != b->device) {
    throw std::runtime_error(
        "Cannot perform arithmetic between tensors on different devices. "
        "Tensor a is on " + a->device.str() + ", tensor b is on " +
        b->device.str() + ". Use .to(), .cuda(), or .cpu() to move tensors.");
  }

  const Device& device = a->device;
  const bool on_cuda = device.is_cuda();
  const std::vector<size_t>& shape = a->shape;
  const bool requires_grad = a->requires_grad || b->requires_grad;

  void (*kernel)(const float*, const float*, float*, const size_t);

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    if (arith_type == ArithType::ADD) kernel = c_add_b_scalar_cuda;
    else if (arith_type == ArithType::SUB) kernel = c_sub_b_scalar_cuda;
    else if (arith_type == ArithType::MUL) kernel = c_mul_b_scalar_cuda;
    else if (arith_type == ArithType::DIV) kernel = c_div_b_scalar_cuda;
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    if (arith_type == ArithType::ADD) kernel = c_add_b_scalar_cpu;
    else if (arith_type == ArithType::SUB) kernel = c_sub_b_scalar_cpu;
    else if (arith_type == ArithType::MUL) kernel = c_mul_b_scalar_cpu;
    else if (arith_type == ArithType::DIV) kernel = c_div_b_scalar_cpu;
  }

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(a->size);
    kernel(a->get(), b->get(), result_cuda, a->size);
    result_tensor = std::make_shared<Tensor>(result_cuda, shape, requires_grad, device);
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[a->size]());
    kernel(a->_data.get(), b->_data.get(), result_data.get(), a->size);
    result_tensor = std::make_shared<Tensor>(result_data, shape, requires_grad);
  }

  if (result_tensor->requires_grad) {
    if (arith_type == ArithType::ADD) {
      result_tensor->grad_fn =
          std::make_shared<AddBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::SUB) {
      result_tensor->grad_fn =
          std::make_shared<SubBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::MUL) {
      result_tensor->grad_fn =
          std::make_shared<MulBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::DIV) {
      result_tensor->grad_fn =
          std::make_shared<DivBackward>(a, b, result_tensor);
    }
  }

  return result_tensor;
}

std::shared_ptr<Tensor> _math_ops_scalars(std::shared_ptr<Tensor>& a,
                                          std::shared_ptr<Tensor>& b,
                                          ArithType arith_type) {
  // Device validation
  if (a->device != b->device) {
    throw std::runtime_error(
        "Cannot perform arithmetic between tensors on different devices. "
        "Tensor a is on " + a->device.str() + ", tensor b is on " +
        b->device.str() + ". Use .to(), .cuda(), or .cpu() to move tensors.");
  }

  const Device& device = a->device;
  const bool on_cuda = device.is_cuda();
  const bool requires_grad = a->requires_grad || b->requires_grad;

  void (*kernel)(const float*, const float*, float*);

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    if (arith_type == ArithType::ADD) kernel = c_add_scalars_cuda;
    else if (arith_type == ArithType::SUB) kernel = c_sub_scalars_cuda;
    else if (arith_type == ArithType::MUL) kernel = c_mul_scalars_cuda;
    else if (arith_type == ArithType::DIV) kernel = c_div_scalars_cuda;
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    if (arith_type == ArithType::ADD) kernel = c_add_scalars_cpu;
    else if (arith_type == ArithType::SUB) kernel = c_sub_scalars_cpu;
    else if (arith_type == ArithType::MUL) kernel = c_mul_scalars_cpu;
    else if (arith_type == ArithType::DIV) kernel = c_div_scalars_cpu;
  }

  std::shared_ptr<Tensor> result_tensor;

  if (on_cuda) {
#ifdef JETDL_WITH_CUDA
    float* result_cuda = cuda::CUDAAllocator::allocate(1);
    kernel(a->get(), b->get(), result_cuda);
    result_tensor = std::make_shared<Tensor>(result_cuda, std::vector<size_t>{}, requires_grad, device);
#endif
  } else {
    auto result_data = std::shared_ptr<float[]>(new float[1]());
    kernel(a->_data.get(), b->_data.get(), result_data.get());
    result_tensor = std::make_shared<Tensor>(result_data, std::vector<size_t>{}, requires_grad);
  }

  if (result_tensor->requires_grad) {
    if (arith_type == ArithType::ADD) {
      result_tensor->grad_fn =
          std::make_shared<AddBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::SUB) {
      result_tensor->grad_fn =
          std::make_shared<SubBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::MUL) {
      result_tensor->grad_fn =
          std::make_shared<MulBackward>(a, b, result_tensor);
    } else if (arith_type == ArithType::DIV) {
      result_tensor->grad_fn =
          std::make_shared<DivBackward>(a, b, result_tensor);
    }
  }

  return result_tensor;
}

}  // namespace jetdl
