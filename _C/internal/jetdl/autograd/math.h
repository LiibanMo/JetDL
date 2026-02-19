#ifndef JETDL_AUTOGRAD_MATH_H
#define JETDL_AUTOGRAD_MATH_H

#include <memory>
#include <vector>

#include "jetdl/autograd.h"

namespace jetdl {

class AddBackward : public Function {
 public:
  AddBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
              std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class SubBackward : public Function {
 public:
  SubBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
              std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class MulBackward : public Function {
 public:
  MulBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
              std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class DivBackward : public Function {
 public:
  DivBackward(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b,
              std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class PowBackward : public Function {
 private:
  int exponent = 0;

 public:
  PowBackward(std::shared_ptr<Tensor>& a, const int exponent,
              std::shared_ptr<Tensor>& result_tensor);

  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class MeanBackward : public Function {
 public:
  MeanBackward(std::shared_ptr<Tensor>& a,
               std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class ExpBackward : public Function {
 public:
  ExpBackward(std::shared_ptr<Tensor>& input,
              std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class LogBackward : public Function {
 private:
  double base_ln = 1.0;  // ln(base), 1.0 for natural log

 public:
  LogBackward(std::shared_ptr<Tensor>& input,
              std::shared_ptr<Tensor>& result_tensor,
              double base_ln = 1.0);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class SinBackward : public Function {
 public:
  SinBackward(std::shared_ptr<Tensor>& input,
              std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class CosBackward : public Function {
 public:
  CosBackward(std::shared_ptr<Tensor>& input,
              std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class TanhBackward : public Function {
 public:
  TanhBackward(std::shared_ptr<Tensor>& input,
               std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class SinhBackward : public Function {
 public:
  SinhBackward(std::shared_ptr<Tensor>& input,
               std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class CoshBackward : public Function {
 public:
  CoshBackward(std::shared_ptr<Tensor>& input,
               std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class AbsBackward : public Function {
 public:
  AbsBackward(std::shared_ptr<Tensor>& input,
              std::shared_ptr<Tensor>& result_tensor);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

class ClampBackward : public Function {
 private:
  float min_val;
  float max_val;

 public:
  ClampBackward(std::shared_ptr<Tensor>& input,
                std::shared_ptr<Tensor>& result_tensor,
                float min_val, float max_val);
  std::vector<std::shared_ptr<Tensor>> apply(
      std::shared_ptr<Tensor>& grad_tensor) override;
};

}  // namespace jetdl

#endif
