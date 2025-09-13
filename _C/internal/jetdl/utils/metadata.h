#ifndef JETDL_UTILS_METADATA_HPP
#define JETDL_UTILS_METADATA_HPP

#include <vector>

namespace jetdl {
namespace utils {

size_t get_size(const std::vector<size_t>& shape);

std::vector<size_t> get_strides(const std::vector<size_t>& shape);

std::vector<size_t> get_byte_strides(const std::vector<size_t>& shape);

}  // namespace utils
}  // namespace jetdl

#endif  // JETDL_UTILS_METADATA_HPP
