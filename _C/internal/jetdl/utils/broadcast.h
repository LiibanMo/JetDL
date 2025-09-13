#ifndef JETDL_UTILS_BROADCAST_HPP
#define JETDL_UTILS_BROADCAST_HPP

#include <cstddef>
#include <vector>

#include "jetdl/utils/auxiliary.h"

namespace jetdl {
namespace utils {

std::pair<std::vector<size_t>, std::vector<size_t>> get_strides(
    const std::vector<size_t>& shapeA, const std::vector<size_t>& shapeB,
    OpType optype);

std::vector<size_t> get_result_shape(const std::vector<size_t>& shapeA,
                                     const std::vector<size_t>& shapeB,
                                     OpType optype);

size_t get_batch_size(const std::vector<size_t>& shape);

}  // namespace utils
}  // namespace jetdl

#endif
