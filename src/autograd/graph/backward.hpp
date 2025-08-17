#pragma once

#include "autograd/function.hpp"
#include "tensor/tensor.hpp"

#include <vector>

enum NodeState {
    VISITED, // Node has left the stack
    VISITING, // Node is in the stack
    UNVISITED // Node has not entered stack yet   
};

std::vector<std::shared_ptr<Function>> topological_sort(std::shared_ptr<Function> node);

namespace autograd {
    void backward(Tensor& input_tensor, Tensor& input_grad);
}