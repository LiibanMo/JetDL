#ifndef JETDL_TENSOR_H
#define JETDL_TENSOR_H

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

namespace jetdl {

class Function;

class Tensor {
 public:
  std::shared_ptr<std::vector<float>> _data = nullptr;
  size_t ndim = 0;
  std::vector<size_t> shape = {};
  size_t size = 0;
  std::vector<size_t> strides = {};
  bool is_contiguous = true;
  bool requires_grad = false;
  std::shared_ptr<Tensor> grad = nullptr;
  std::shared_ptr<Function> grad_fn = nullptr;

  Tensor();

  Tensor(const py::object& data, const bool requires_grad = false);

  Tensor(const std::shared_ptr<std::vector<float>>& data,
         const std::vector<size_t>& shape, const bool requires_grad = false);

  Tensor(const float& data, const bool requires_grad = false);

  Tensor(const Tensor& other);

  std::shared_ptr<Tensor> view(const std::vector<size_t>& shape) const;

  Tensor operator=(const Tensor& other);

  ~Tensor() = default;
};

Tensor tensor(const py::object& data, const bool requires_grad);

}  // namespace jetdl

#endif
