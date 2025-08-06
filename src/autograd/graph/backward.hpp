#pragma once

#include "autograd/function.hpp"
#include "tensor/tensor.hpp"

#include <vector>

enum NodeState {
    VISITED, // Node has left the stack
    VISITING, // Node is in the stack
    UNVISITED // Node has not entered stack yet   
};

std::vector<Function> topological_sort(const Function& node);

namespace autograd {
    void backward(const Tensor& input_grad);
}