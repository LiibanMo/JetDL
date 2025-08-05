#pragma once

#include <memory>
#include "tensor/tensor.hpp"

class Function {
public:
    virtual ~Function() = default;
    virtual Tensor apply() = 0;
    std::shared_ptr<Function> next_function;
};
