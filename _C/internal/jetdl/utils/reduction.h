#ifndef JETDL_UTILS_REDUCTION_HPP
#define JETDL_UTILS_REDUCTION_HPP

#include <cstddef>
#include <vector>

namespace jetdl {
namespace utils {

std::vector<size_t> get_shape(const std::vector<size_t>& shape,
                              const std::vector<size_t>& axes);

std::vector<size_t> get_dest_strides(const std::vector<size_t>& original_shape,
                                     const std::vector<size_t>& result_strides,
                                     const std::vector<size_t>& axes);

}  // namespace utils
}  // namespace jetdl

#endif
