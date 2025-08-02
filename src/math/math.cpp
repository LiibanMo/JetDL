#include "math.hpp"
#include "ops/ops.hpp"

namespace math {    
    
    Tensor add(const Tensor& a, const Tensor& b) {
        const std::string op = "ADD";
        if (a.ndim > 0 && b.ndim > 0) {
            return c_ops(a, b, op);
        } else if (a.ndim == 0 && b.ndim > 0) {
            return c_ops_scalar_a(a._data[0], b, op);
        } else if (a.ndim > 0 && b.ndim == 0) {
            return c_ops_scalar_b(a, b._data[0], op);
        } else {
            return c_ops_scalars(a, b, op);
        }
    }

    Tensor sub(const Tensor& a, const Tensor& b) {
        const std::string op = "SUB";
        if (a.ndim > 0 && b.ndim > 0) {
            return c_ops(a, b, op);
        } else if (a.ndim == 0 && b.ndim > 0) {
            return c_ops_scalar_a(a._data[0], b, op);
        } else if (a.ndim > 0 && b.ndim == 0) {
            return c_ops_scalar_b(a, b._data[0], op);
        } else {
            return c_ops_scalars(a, b, op);
        }
    }

    Tensor mul(const Tensor& a, const Tensor& b) {
        const std::string op = "MUL";
        if (a.ndim > 0 && b.ndim > 0) {
            return c_ops(a, b, op);
        } else if (a.ndim == 0 && b.ndim > 0) {
            return c_ops_scalar_a(a._data[0], b, op);
        } else if (a.ndim > 0 && b.ndim == 0) {
            return c_ops_scalar_b(a, b._data[0], op);
        } else {
            return c_ops_scalars(a, b, op);
        }
    }

    Tensor div(const Tensor& a, const Tensor& b) {
        const std::string op = "DIV";
        if (a.ndim > 0 && b.ndim > 0) {
            return c_ops(a, b, op);
        } else if (a.ndim == 0 && b.ndim > 0) {
            return c_ops_scalar_a(a._data[0], b, op);
        } else if (a.ndim > 0 && b.ndim == 0) {
            return c_ops_scalar_b(a, b._data[0], op);
        } else {
            return c_ops_scalars(a, b, op);
        }
    }
    
}