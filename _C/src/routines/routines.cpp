#include "jetdl/routines.h"
#include "jetdl/C/routines/creation.h"

std::unique_ptr<Tensor, TensorDeleter>
routines_ones(const std::vector<size_t> shape) {
  Tensor *result_tensor = c_routines_ones(shape.data(), shape.size());
  return std::unique_ptr<Tensor, TensorDeleter>(result_tensor);
}
