#include <cmath>
#include <memory>
#include <vector>

#include "jetdl/autograd/math.h"
#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

// ---- ExpBackward ----

ExpBackward::ExpBackward(std::shared_ptr<Tensor>& input,
                         std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> ExpBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[exp(x)] = exp(x) = output (this->tensor)
    auto output = this->tensor.lock();
    grads[0] = math::mul(grad_tensor, output);
  }

  return grads;
}

// ---- LogBackward ----

LogBackward::LogBackward(std::shared_ptr<Tensor>& input,
                         std::shared_ptr<Tensor>& result_tensor,
                         double base_ln) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
  this->base_ln = base_ln;
}

std::vector<std::shared_ptr<Tensor>> LogBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[log_b(x)] = 1 / (x * ln(b))
    auto denom_data = std::shared_ptr<float[]>(new float[input->size]());
    const float base = static_cast<float>(this->base_ln);
    for (size_t i = 0; i < input->size; i++) {
      denom_data[i] = input->_data[i] * base;
    }
    auto denom = std::make_shared<Tensor>(denom_data, input->shape, false);
    grads[0] = math::div(grad_tensor, denom);
  }

  return grads;
}

// ---- SinBackward ----

SinBackward::SinBackward(std::shared_ptr<Tensor>& input,
                         std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> SinBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[sin(x)] = cos(x)
    auto cos_data = std::shared_ptr<float[]>(new float[input->size]());
    for (size_t i = 0; i < input->size; i++) {
      cos_data[i] = std::cos(input->_data[i]);
    }
    auto cos_tensor = std::make_shared<Tensor>(cos_data, input->shape, false);
    grads[0] = math::mul(grad_tensor, cos_tensor);
  }

  return grads;
}

// ---- CosBackward ----

CosBackward::CosBackward(std::shared_ptr<Tensor>& input,
                         std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> CosBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[cos(x)] = -sin(x)
    auto neg_sin_data = std::shared_ptr<float[]>(new float[input->size]());
    for (size_t i = 0; i < input->size; i++) {
      neg_sin_data[i] = -std::sin(input->_data[i]);
    }
    auto neg_sin = std::make_shared<Tensor>(neg_sin_data, input->shape, false);
    grads[0] = math::mul(grad_tensor, neg_sin);
  }

  return grads;
}

// ---- TanhBackward ----

TanhBackward::TanhBackward(std::shared_ptr<Tensor>& input,
                           std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> TanhBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[tanh(x)] = 1 - tanh²(x) = 1 - output²
    auto output = this->tensor.lock();
    const float* output_data = output->_data.get();
    auto sech2_data = std::shared_ptr<float[]>(new float[input->size]());
    for (size_t i = 0; i < input->size; i++) {
      sech2_data[i] = 1.0f - output_data[i] * output_data[i];
    }
    auto sech2 = std::make_shared<Tensor>(sech2_data, input->shape, false);
    grads[0] = math::mul(grad_tensor, sech2);
  }

  return grads;
}

// ---- SinhBackward ----

SinhBackward::SinhBackward(std::shared_ptr<Tensor>& input,
                           std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> SinhBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[sinh(x)] = cosh(x)
    auto cosh_data = std::shared_ptr<float[]>(new float[input->size]());
    for (size_t i = 0; i < input->size; i++) {
      cosh_data[i] = std::cosh(input->_data[i]);
    }
    auto cosh_tensor = std::make_shared<Tensor>(cosh_data, input->shape, false);
    grads[0] = math::mul(grad_tensor, cosh_tensor);
  }

  return grads;
}

// ---- CoshBackward ----

CoshBackward::CoshBackward(std::shared_ptr<Tensor>& input,
                           std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> CoshBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[cosh(x)] = sinh(x)
    auto sinh_data = std::shared_ptr<float[]>(new float[input->size]());
    for (size_t i = 0; i < input->size; i++) {
      sinh_data[i] = std::sinh(input->_data[i]);
    }
    auto sinh_tensor = std::make_shared<Tensor>(sinh_data, input->shape, false);
    grads[0] = math::mul(grad_tensor, sinh_tensor);
  }

  return grads;
}

// ---- AbsBackward ----

AbsBackward::AbsBackward(std::shared_ptr<Tensor>& input,
                         std::shared_ptr<Tensor>& result_tensor) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
}

std::vector<std::shared_ptr<Tensor>> AbsBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // d/dx[|x|] = sign(x)
    auto sign_data = std::shared_ptr<float[]>(new float[input->size]());
    for (size_t i = 0; i < input->size; i++) {
      const float x = input->_data[i];
      sign_data[i] = (x > 0.0f) ? 1.0f : (x < 0.0f) ? -1.0f : 0.0f;
    }
    auto sign_tensor = std::make_shared<Tensor>(sign_data, input->shape, false);
    grads[0] = math::mul(grad_tensor, sign_tensor);
  }

  return grads;
}

// ---- ClampBackward ----

ClampBackward::ClampBackward(std::shared_ptr<Tensor>& input,
                             std::shared_ptr<Tensor>& result_tensor,
                             float min_val, float max_val) {
  this->next_functions = std::vector<std::shared_ptr<Function>>{input->grad_fn};
  this->saved_tensors = std::vector<std::shared_ptr<Tensor>>{input};
  this->tensor = result_tensor;
  this->min_val = min_val;
  this->max_val = max_val;
}

std::vector<std::shared_ptr<Tensor>> ClampBackward::apply(
    std::shared_ptr<Tensor>& grad_tensor) {
  std::shared_ptr<Tensor>& input = this->saved_tensors[0];
  auto grads = std::vector<std::shared_ptr<Tensor>>(1, nullptr);

  if (input->requires_grad) {
    // gradient is 1 where input is within [min_val, max_val], 0 outside
    auto mask_data = std::shared_ptr<float[]>(new float[input->size]());
    for (size_t i = 0; i < input->size; i++) {
      const float x = input->_data[i];
      mask_data[i] = (x >= this->min_val && x <= this->max_val) ? 1.0f : 0.0f;
    }
    auto mask = std::make_shared<Tensor>(mask_data, input->shape, false);
    grads[0] = math::mul(grad_tensor, mask);
  }

  return grads;
}

}  // namespace jetdl
