#ifndef AUTOGRAD_FUNCTION_H
#define AUTOGRAD_FUNCTION_H

#include "tensor/tensor.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Function Function;

struct Function {
  Tensor *prev_tensors;
  size_t nprev_tensors;
  Tensor *current_grad;
  void (*apply)(const Function *fn);
};

#ifdef __cplusplus
}
#endif
#endif
