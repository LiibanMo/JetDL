#ifndef JETDL_TENSOR_H
#define JETDL_TENSOR_H

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

namespace jetdl {

class Function;

class Tensor : public std::enable_shared_from_this<Tensor> {
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

  Tensor operator=(const Tensor& other);
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator-() const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  std::shared_ptr<Tensor> view(const std::vector<size_t>& shape) const;
  std::shared_ptr<Tensor> squeeze(const size_t axis) const;
  std::shared_ptr<Tensor> unsqueeze(const size_t axis) const;

  ~Tensor() = default;
};

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor>& a,
                                  std::shared_ptr<Tensor>& b);

Tensor tensor(const py::object& data, const bool requires_grad);

}  // namespace jetdl

#endif
