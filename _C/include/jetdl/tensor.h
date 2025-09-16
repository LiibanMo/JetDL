#ifndef JETDL_TENSOR_H
#define JETDL_TENSOR_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

class Function;

namespace jetdl {

class Tensor {
 public:
  std::shared_ptr<std::vector<float>> _data;
  size_t ndim;
  std::vector<size_t> shape;
  size_t size;
  std::vector<size_t> strides;
  bool is_contiguous;
  bool requires_grad;
  std::shared_ptr<Function> grad_fn;
  std::shared_ptr<Tensor> grad;

  Tensor();

  Tensor(const py::object& data, const bool requires_grad = false);

  Tensor(const std::shared_ptr<std::vector<float>>& data,
         const std::vector<size_t>& shape, const bool requires_grad = false);

  Tensor(const float& data, const bool requires_grad = false);

  Tensor(const Tensor& other);

  Tensor view(const std::vector<size_t>& shape) const;

  Tensor operator=(const Tensor& other);

  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  Tensor matmul(const Tensor& other) const;
  Tensor T() const;
  Tensor mT() const;

  Tensor sum(const std::vector<int>& axes) const;
};

Tensor tensor(const py::object& data, const bool requires_grad);

}  // namespace jetdl

#endif
