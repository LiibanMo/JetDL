#include "ops.hpp"
#include "ops/add.hpp"

namespace linalg {

    Tensor add(const Tensor& a, const Tensor& b) {
        return c_add_broadcasted(a, b);
    }
    
}