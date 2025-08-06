#include "backward.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stack>
#include <stdexcept>
#include <unordered_map>

namespace std {
    template <>
    struct hash<Function> {
        std::size_t operator()(const Function& fn) const {
            return std::hash<std::shared_ptr<void>>()(fn._unique_identity_ptr);
        }
    };
}

std::vector<Function> topological_sort(Function& node) {
    std::vector<Function> graph = {};
    std::unordered_map<Function, NodeState> node_states;
    std::stack<Function> stack;
    stack.push(node);
    
    while (!stack.empty()) {
        Function current_node = stack.top();
        node_states[current_node] = VISITING;
        
        Function unvisited_child;
        bool all_child_node_visited = true;

        for (Function& fn : current_node.next_function) {
            if (node_states.find(fn) == node_states.end()) {
                all_child_node_visited = false;
                unvisited_child = fn;
                break;
            } else {
                if (node_states[fn] == VISITING) {
                    throw std::runtime_error("cycle detected in computing graph.");
                }
            }
        }

        if (all_child_node_visited) {
            stack.pop();
            node_states[current_node] = VISITED;
            graph.push_back(current_node);
        } else {
            stack.push(unvisited_child);
        }
    }

    std::reverse(graph.begin(), graph.end());

    return graph;
}

void autograd::backward(const Tensor& input_grad) {
    if (input_grad.grad_fn) {
        const std::vector<Function> graph = topological_sort(*input_grad.grad_fn);
    }
}