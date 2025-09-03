#include "routines/routines.h"
#include "routines/creation/ones.h"
#include "tensor/python/bindings.h"
#include "tensor/tensor.h"
#include <memory>

std::unique_ptr<Tensor, TensorDeleter>
routines_ones(const std::vector<size_t> shape) {
  Tensor *result_tensor = c_routines_ones(shape.data(), shape.size());
  return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
}
