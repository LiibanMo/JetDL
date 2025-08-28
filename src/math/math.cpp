#include "math.h"
#include "math/functions/reduction.h"
#include "math/ops/arith.h"
#include "python/utils/check.h"
#include "utils/auxiliary.h"

#include <stdlib.h>
#include <stdexcept>

Tensor& math_ops(const Tensor& a, const Tensor& b, const std::string op) {
    utils_check_ops_shapes(a.shape, a.ndim, b.shape, b.ndim);
    if (op == "ADD") {
        return *c_math_ops(&a, &b, ADD);
    } else if (op == "SUB") {
        return *c_math_ops(&a, &b, SUB);
    } else if (op == "MUL") {
        return *c_math_ops(&a, &b, MUL);
    } else if (op == "DIV") {
        return *c_math_ops(&a, &b, DIV);
    } else {
        throw std::logic_error("ops can only of the following options: ADD, SUB, MUL, DIV.");
    }
}

Tensor& math_sum(const Tensor& a, std::vector<int>& axes) {
    if (axes.empty()) {
        return *c_math_total_sum(&a);
    } else {
        utils_check_axes(a.shape, a.ndim, axes.data(), axes.size());
        return *c_math_sum_over_axes(&a, axes.data(), axes.size());
    }
}