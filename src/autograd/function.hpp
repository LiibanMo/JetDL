#pragma once

#include <memory>
#include <vector>

class Tensor; // forward declaration

class Function {
    public:
        std::shared_ptr<Tensor> operandA = nullptr;
        std::shared_ptr<Tensor> operandB = nullptr;

        std::shared_ptr<void> _unique_identity_ptr;

        std::vector<std::shared_ptr<Function>> next_function;

        Function();
        virtual ~Function() = default;

        bool operator==(const Function& other) const {
            return this->_unique_identity_ptr == other._unique_identity_ptr;
        }

        virtual void apply(std::shared_ptr<Tensor> incoming_gradient) = 0;
};