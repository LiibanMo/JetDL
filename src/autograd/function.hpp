#pragma once

#include <memory>
#include <vector>

class Tensor; // forward declaration

class Function {
    public:
        std::shared_ptr<void> _unique_identity_ptr;
        std::vector<Function> next_function;

        bool operator==(const Function& other) const {
            return this->_unique_identity_ptr == other._unique_identity_ptr;
        }
};

class Grad {
    public:
        std::shared_ptr<Tensor> gradA;
        std::shared_ptr<Tensor> gradB;
};