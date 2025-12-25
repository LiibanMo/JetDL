#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/device.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/metadata.h"
#include "jetdl/utils/py.h"

#ifdef JETDL_WITH_CUDA
#include <cuda_runtime.h>

#include "jetdl/cuda/allocator.h"
#endif

namespace py = pybind11;

namespace jetdl {

Tensor::Tensor()
    : _data(nullptr),
      _cuda_data(nullptr),
      device(Device::cpu()),
      ndim(0),
      shape({}),
      size(0),
      strides({}),
      is_contiguous(true),
      requires_grad(false),
      grad_fn(nullptr),
      grad(nullptr) {}

Tensor::Tensor(const py::object& data, const bool requires_grad,
               const Device& device)
    : _cuda_data(nullptr),
      device(device),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(nullptr),
      is_contiguous(true) {
  // First, parse data into CPU memory
  std::shared_ptr<float[]> cpu_data;

  if (py::isinstance<py::list>(data)) {
    jetdl::utils::py_check_data_consistency(data);
    this->ndim = jetdl::utils::py_get_ndim(data);
    this->shape = jetdl::utils::py_get_shape(data, this->ndim);
    this->size = jetdl::utils::get_size(this->shape);
    this->strides = jetdl::utils::get_strides(this->shape);
    cpu_data = jetdl::utils::py_flatten_list(data);
  } else if (py::isinstance<py::int_>(data) ||
             py::isinstance<py::float_>(data)) {
    this->ndim = 0;
    this->shape = {};
    this->size = 1;
    this->strides = {};
    cpu_data = std::shared_ptr<float[]>(new float[1]());
    cpu_data[0] = py::cast<float>(data);
  } else {
    throw py::type_error(
        py::str("init(): type '{}' invalid").format(py::type::of(data)));
  }

  // Now place data on the target device
  if (device.is_cuda()) {
#ifdef JETDL_WITH_CUDA
    if (!cuda_is_available()) {
      throw std::runtime_error(
          "CUDA is not available. Cannot create tensor on CUDA device.");
    }
    // Allocate GPU memory and copy data
    this->_cuda_data = cuda::CUDAAllocator::allocate(this->size);
    cudaMemcpy(this->_cuda_data, cpu_data.get(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    // CPU data is not needed, let shared_ptr clean it up
    this->_data = nullptr;
#else
    throw std::runtime_error(
        "JetDL was compiled without CUDA support. "
        "Cannot create tensor on CUDA device.");
#endif
  } else {
    // CPU tensor - just use the parsed data
    this->_data = cpu_data;
  }
}

Tensor::Tensor(const std::shared_ptr<float[]>& data,
               const std::vector<size_t>& shape, const bool requires_grad,
               const Device& device)
    : _data(data),
      _cuda_data(nullptr),
      device(device),
      ndim(shape.size()),
      shape(shape),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(nullptr),
      is_contiguous(true) {
  // This constructor is for CPU data passed via shared_ptr
  // CUDA data should use the cuda_data constructor
  if (device.is_cuda()) {
    throw std::runtime_error(
        "Cannot create CUDA tensor from CPU shared_ptr. "
        "Use the appropriate CUDA constructor.");
  }
  this->size = jetdl::utils::get_size(this->shape);
  this->strides = jetdl::utils::get_strides(this->shape);
}

Tensor::Tensor(float* cuda_data, const std::vector<size_t>& shape,
               const bool requires_grad, const Device& device)
    : _data(nullptr),
      _cuda_data(cuda_data),
      device(device),
      ndim(shape.size()),
      shape(shape),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(nullptr),
      is_contiguous(true) {
  // This constructor is for CUDA tensors with pre-allocated device memory
  // Takes ownership of cuda_data - will be freed in destructor
  if (!device.is_cuda()) {
    throw std::runtime_error(
        "Cannot use raw pointer constructor for CPU tensor. "
        "Use the shared_ptr constructor instead.");
  }
  this->size = jetdl::utils::get_size(this->shape);
  this->strides = jetdl::utils::get_strides(this->shape);
}

Tensor::Tensor(const float& data, const bool requires_grad,
               const Device& device)
    : _cuda_data(nullptr),
      device(device),
      ndim(0),
      shape({}),
      size(1),
      strides({}),
      is_contiguous(true),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(nullptr) {
  if (device.is_cuda()) {
#ifdef JETDL_WITH_CUDA
    if (!cuda_is_available()) {
      throw std::runtime_error(
          "CUDA is not available. Cannot create tensor on CUDA device.");
    }
    // Allocate GPU memory for scalar
    this->_cuda_data = cuda::CUDAAllocator::allocate(1);
    cudaMemcpy(this->_cuda_data, &data, sizeof(float), cudaMemcpyHostToDevice);
#else
    throw std::runtime_error(
        "JetDL was compiled without CUDA support. "
        "Cannot create tensor on CUDA device.");
#endif
  } else {
    this->_data = std::shared_ptr<float[]>(new float[1]());
    this->_data[0] = data;
  }
}

Tensor::Tensor(const Tensor& other, const bool requires_grad)
    : _data(other._data),
      _cuda_data(other._cuda_data),
      device(other.device),
      ndim(other.ndim),
      shape(other.shape),
      size(other.size),
      strides(other.strides),
      is_contiguous(other.is_contiguous),
      requires_grad(requires_grad),
      grad_fn(other.grad_fn),
      grad(other.grad) {}

Tensor::Tensor(const Tensor& other)
    : _data(other._data),
      _cuda_data(other._cuda_data),
      device(other.device),
      ndim(other.ndim),
      shape(other.shape),
      size(other.size),
      strides(other.strides),
      is_contiguous(other.is_contiguous),
      requires_grad(other.requires_grad),
      grad_fn(other.grad_fn),
      grad(other.grad) {}

Tensor::~Tensor() {
  // Free CUDA memory if this tensor owns CUDA data
  if (_cuda_data != nullptr) {
#ifdef JETDL_WITH_CUDA
    cuda::CUDAAllocator::deallocate(_cuda_data);
#endif
    _cuda_data = nullptr;
  }
}

}  // namespace jetdl
