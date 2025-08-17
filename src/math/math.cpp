#include "math.hpp"
#include "ops/ops.hpp"
#include "function/reduction.hpp"

#include <vector>

namespace math {    

    namespace ops {

        Tensor add(Tensor& a, Tensor& b) {
            const std::string op = "ADD";
            if (a.ndim > 0 && b.ndim > 0) {
                return c_ops(a, b, op);
            } else if (a.ndim == 0 && b.ndim > 0) {
                return c_ops_scalar_a(a, b, op);
            } else if (a.ndim > 0 && b.ndim == 0) {
                return c_ops_scalar_b(a, b, op);
            } else {
                return c_ops_scalars(a, b, op);
            }
        }

        Tensor sub(Tensor& a, Tensor& b) {
            const std::string op = "SUB";
            if (a.ndim > 0 && b.ndim > 0) {
                return c_ops(a, b, op);
            } else if (a.ndim == 0 && b.ndim > 0) {
                return c_ops_scalar_a(a, b, op);
            } else if (a.ndim > 0 && b.ndim == 0) {
                return c_ops_scalar_b(a, b, op);
            } else {
                return c_ops_scalars(a, b, op);
            }
        }

        Tensor mul(Tensor& a, Tensor& b) {
            const std::string op = "MUL";
            if (a.ndim > 0 && b.ndim > 0) {
                return c_ops(a, b, op);
            } else if (a.ndim == 0 && b.ndim > 0) {
                return c_ops_scalar_a(a, b, op);
            } else if (a.ndim > 0 && b.ndim == 0) {
                return c_ops_scalar_b(a, b, op);
            } else {
                return c_ops_scalars(a, b, op);
            }
        }

        Tensor div(Tensor& a, Tensor& b) {
            const std::string op = "DIV";
            if (a.ndim > 0 && b.ndim > 0) {
                return c_ops(a, b, op);
            } else if (a.ndim == 0 && b.ndim > 0) {
                return c_ops_scalar_a(a, b, op);
            } else if (a.ndim > 0 && b.ndim == 0) {
                return c_ops_scalar_b(a, b, op);
            } else {
                return c_ops_scalars(a, b, op);
            }
        }
    
    }

    namespace function {

        Tensor total_sum(const Tensor& tensor) {
            return c_total_sum(tensor);
        }

        Tensor sum_w_axes(const Tensor& tensor, const std::vector<int>& axes) {
            return c_sum_w_axes(tensor, axes);
        }

    }
}