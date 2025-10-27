#ifndef JETDL_TENSOR_H
#define JETDL_TENSOR_H

#include <pybind11/pybind11.h>

#include <memory>
#include <vector>

namespace py = pybind11;

namespace jetdl {

class Function;

enum class DType {};

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::shared_ptr<float[]> _data;
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
  Tensor(const std::shared_ptr<float[]>& data, const std::vector<size_t>& shape,
         const bool requires_grad = false);
  Tensor(const float& data, const bool requires_grad = false);
  Tensor(const Tensor& other, const bool requires_grad);
  Tensor(const Tensor& other);

  Tensor operator=(const Tensor& other);
  Tensor operator+(Tensor& other);
  Tensor operator-(Tensor& other);
  Tensor operator-();
  Tensor operator*(Tensor& other);
  Tensor operator/(Tensor& other);

  float* get() { return this->_data.get(); }

  virtual ~Tensor() = default;
};

Tensor tensor(const py::object& data, const bool requires_grad);

}  // namespace jetdl

#endif
