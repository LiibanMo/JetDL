#ifndef JETDL_UTILS_AUXILIARY_HPP
#define JETDL_UTILS_AUXILIARY_HPP

#include <cstddef>
#include <vector>

namespace jetdl {
namespace utils {

enum class OpType { MATMUL, DOT, ARITHMETIC, REDUCTION };

enum class ArithType { ADD, SUB, MUL, DIV };

inline long get_next_multiple(const long start, const long factor) {
  return ((start + factor - 1) / factor) * factor;
}

std::vector<size_t> populate_linear_idxs(const std::vector<size_t>& shape,
                                         const std::vector<size_t>& strides,
                                         OpType optype);

std::vector<size_t> make_axes_positive(const std::vector<int>& axes,
                                       size_t ndim);

}  // namespace utils
}  // namespace jetdl

#endif
