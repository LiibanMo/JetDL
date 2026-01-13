#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef JETDL_WITH_OPENMP
#include <omp.h>
#endif

#include "jetdl/autograd/linalg.h"
#include "jetdl/linalg/kernel.h"
#include "jetdl/linalg/product.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"
#include "jetdl/utils/auxiliary.h"
#include "jetdl/utils/broadcast.h"
#include "jetdl/utils/metadata.h"

#ifdef JETDL_WITH_CUDA
#include "jetdl/cuda/allocator.h"
#endif

namespace jetdl {

std::shared_ptr<Tensor> _linalg_vecmat(std::shared_ptr<Tensor>& a,
                                       std::shared_ptr<Tensor>& b) {
  std::vector<size_t> view_shape_a = {1};
  view_shape_a.push_back(a->shape[0]);

  std::shared_ptr<Tensor> view_tensor_a = view(a, view_shape_a);

  std::shared_ptr<Tensor> view_result_tensor = _linalg_matmul(view_tensor_a, b);

  const std::vector<size_t>& shape =
      utils::get_result_shape(a->shape, b->shape, utils::OpType::MATMUL);

  std::shared_ptr<Tensor> result_tensor = view(view_result_tensor, shape);
  result_tensor->requires_grad = a->requires_grad || b->requires_grad;

  return result_tensor;
}

}  // namespace jetdl