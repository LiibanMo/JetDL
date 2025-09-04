#ifndef JETDL_TENSOR_H
#define JETDL_TENSOR_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float *_data;
  size_t ndim;
  size_t *shape;
  size_t size;
  size_t *strides;
  bool is_contiguous;

  bool requires_grad;
  void *grad_fn;
} Tensor;

Tensor *create_tensor(float *_data, size_t *shape, const size_t ndim);
Tensor *copy_tensor(const Tensor *src, Tensor *dest);
void destroy_tensor(Tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
