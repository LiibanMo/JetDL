#ifndef JETDL_UTILS_CHECK_HPP
#define JETDL_UTILS_CHECK_HPP

#include <vector>

#include "jetdl/utils/auxiliary.h"

class Tensor;

namespace jetdl {
namespace utils {

void check_axes(const std::vector<size_t>& shape, const std::vector<int>& axes,
                SubModule submodule = SubModule::MATH);
void check_ops_shapes(const std::vector<size_t>& shapeA,
                      const std::vector<size_t>& shapeB);
void check_dot_shapes(const std::vector<size_t>& shapeA,
                      const std::vector<size_t>& shapeB);
void check_vecmat_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB);
void check_matvec_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB);
void check_matmul_shapes(const std::vector<size_t>& shapeA,
                         const std::vector<size_t>& shapeB);

}  // namespace utils
}  // namespace jetdl

#endif
