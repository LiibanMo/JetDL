#include "math.hpp"
#include "ops.hpp"

namespace math {    
    
    Tensor add(const Tensor& a, const Tensor& b) {
        return c_ops(a, b, "ADD");
    }
    
    Tensor sub(const Tensor& a, const Tensor& b) {
        return c_ops(a, b, "SUB");
    }
    
    Tensor mul(const Tensor& a, const Tensor& b) {
        return c_ops(a, b, "MUL");
    }
    
    Tensor div(const Tensor& a, const Tensor& b) {
        return c_ops(a, b, "DIV");
    }
    
}