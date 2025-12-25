#ifndef JETDL_TENSOR_H
#define JETDL_TENSOR_H

#include <pybind11/pybind11.h>

#include <memory>
#include <vector>

#include "jetdl/device.h"

namespace py = pybind11;

namespace jetdl {

class Function;

enum class DType {};

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  // Data storage
  std::shared_ptr<float[]> _data;  // CPU data (non-null when device=CPU)
  float* _cuda_data = nullptr;     // CUDA data (non-null when device=CUDA)

  // Device tracking
  Device device;

  // Metadata
  size_t ndim = 0;
  std::vector<size_t> shape = {};
  size_t size = 0;
  std::vector<size_t> strides = {};
  bool is_contiguous = true;

  // Autograd
  bool requires_grad = false;
  std::shared_ptr<Tensor> grad = nullptr;
  std::shared_ptr<Function> grad_fn = nullptr;

  Tensor();
  Tensor(const py::object& data, const bool requires_grad = false,
         const Device& device = Device::cpu());
  Tensor(const std::shared_ptr<float[]>& data, const std::vector<size_t>& shape,
         const bool requires_grad = false,
         const Device& device = Device::cpu());
  // Constructor for CUDA tensors with pre-allocated device memory
  // Takes ownership of the cuda_data pointer
  Tensor(float* cuda_data, const std::vector<size_t>& shape,
         const bool requires_grad, const Device& device);
  Tensor(const float& data, const bool requires_grad = false,
         const Device& device = Device::cpu());
  Tensor(const Tensor& other, const bool requires_grad);
  Tensor(const Tensor& other);

  Tensor operator=(const Tensor& other);
  Tensor operator+(Tensor& other);
  Tensor operator-(Tensor& other);
  Tensor operator-();
  Tensor operator*(Tensor& other);
  Tensor operator/(Tensor& other);

  // Returns pointer to data on the tensor's device
  float* get() {
    if (device.is_cuda()) {
      return _cuda_data;
    }
    return _data.get();
  }

  // Device transfer methods
  std::shared_ptr<Tensor> to(const Device& target) const;
  std::shared_ptr<Tensor> cuda(int device_id = 0) const;
  std::shared_ptr<Tensor> cpu() const;

  // Device check methods
  bool is_cuda() const { return device.is_cuda(); }
  bool is_cpu() const { return device.is_cpu(); }

  ~Tensor();
};

Tensor tensor(const py::object& data, const bool requires_grad);

}  // namespace jetdl

#endif
