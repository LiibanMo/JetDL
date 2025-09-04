#include "jetdl/autograd.h"
#include "jetdl/tensor.h"

struct Function {
  Tensor *prev_tensors;
  size_t nprev_tensors;
  Tensor *grad;
  void (*apply)(const Function *fn);
};
