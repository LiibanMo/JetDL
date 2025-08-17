#include "ops.hpp"
#include "tensor/tensor.hpp"
#include "math/function/reduction.hpp"
#include <cstddef>

AddBackward::AddBackward(Tensor& a, Tensor& b) {
    this->operandA = std::shared_ptr<Tensor>(&a);
    this->operandB = std::shared_ptr<Tensor>(&b);
}

void AddBackward::apply(std::shared_ptr<Tensor> incoming_gradient) {
    this->operandA->grad = std::shared_ptr<Tensor>(); 
    this->operandB->grad = std::shared_ptr<Tensor>();

    if (this->operandA->ndim == incoming_gradient->ndim) {
        *this->operandA->grad = incoming_gradient->copy();
    } else if (this->operandA->ndim < incoming_gradient->ndim) {
        *this->operandA->grad = c_sum_to_size(*incoming_gradient, this->operandA->shape);
    }

    if (this->operandB->ndim == incoming_gradient->ndim) {
        *this->operandB->grad = incoming_gradient->copy();
    } else if (this->operandB->ndim < incoming_gradient->ndim) {
        *this->operandB->grad = c_sum_to_size(*incoming_gradient, this->operandB->shape);
    }
}

SubBackward::SubBackward(Tensor& a, Tensor& b) {
    this->operandA = std::shared_ptr<Tensor>(&a);
    this->operandB = std::shared_ptr<Tensor>(&b);
}

void SubBackward::apply(std::shared_ptr<Tensor> incoming_gradient) {
    NULL;
}

MulBackward::MulBackward(Tensor& a, Tensor& b) {
    this->operandA = std::shared_ptr<Tensor>(&a);
    this->operandB = std::shared_ptr<Tensor>(&b);
}

void MulBackward::apply(std::shared_ptr<Tensor> incoming_gradient) {
    NULL;
}

DivBackward::DivBackward(Tensor& a, Tensor& b) {
    this->operandA = std::shared_ptr<Tensor>(&a);
    this->operandB = std::shared_ptr<Tensor>(&b);
}

void DivBackward::apply(std::shared_ptr<Tensor> incoming_gradient) {
    NULL;
}