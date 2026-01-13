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

std::shared_ptr<Tensor> _linalg_dot(std::shared_ptr<Tensor>& a,
                                    std::shared_ptr<Tensor>& b) {
  std::vector<size_t> view_shape_a = {1};
  view_shape_a.push_back(a->shape[0]);

  std::vector<size_t> view_shape_b = b->shape;
  view_shape_b.push_back(1);

  std::shared_ptr<Tensor> view_tensor_a = view(a, view_shape_a);
  std::shared_ptr<Tensor> view_tensor_b = view(b, view_shape_b);

  std::shared_ptr<Tensor> view_result_tensor =
      _linalg_matmul(view_tensor_a, view_tensor_b);

  const Device& device = a->device;
  const bool requires_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> result_tensor;

  if (device.is_cuda()) {
#ifdef JETDL_WITH_CUDA
    // For CUDA, reshape the (1,1) result to scalar shape
    // Reuse the existing CUDA data by creating a new tensor with scalar shape
    result_tensor = std::make_shared<Tensor>();
    result_tensor->_cuda_data = view_result_tensor->_cuda_data;
    result_tensor->_data = nullptr;
    result_tensor->device = device;
    result_tensor->ndim = 0;
    result_tensor->shape = {};
    result_tensor->size = 1;
    result_tensor->strides = {};
    result_tensor->is_contiguous = true;
    result_tensor->requires_grad = requires_grad;
    // Prevent double-free by nulling out source's cuda_data
    view_result_tensor->_cuda_data = nullptr;
#else
    throw std::runtime_error("JetDL compiled without CUDA support");
#endif
  } else {
    result_tensor =
        std::make_shared<Tensor>(view_result_tensor->_data[0], requires_grad);
  }

  if (result_tensor->requires_grad) {
    result_tensor->grad_fn = std::make_shared<DotBackward>(a, b, result_tensor);
  }

  return result_tensor;
}

}  // namespace jetdl