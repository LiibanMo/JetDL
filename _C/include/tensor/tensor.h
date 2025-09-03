#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tensor Tensor;

struct Tensor {
  float *_data;
  size_t ndim;
  size_t *shape;
  size_t size;
  size_t *strides;
  bool is_contiguous;

  bool requires_grad;
  Tensor *grad;
  void *grad_fn;
};

Tensor *create_tensor(float *_data, size_t *shape, const size_t ndim);
Tensor *copy_tensor(const Tensor *src, Tensor *dest);
void destroy_tensor(Tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
