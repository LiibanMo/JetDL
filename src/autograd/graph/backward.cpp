#include "backward.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stack>
#include <stdexcept>
#include <unordered_map>

#include <iostream> // DEV

std::vector<std::shared_ptr<Function>> topological_sort(std::shared_ptr<Function> node) {
    std::vector<std::shared_ptr<Function>> graph = {};
    std::unordered_map<std::shared_ptr<Function>, NodeState> node_states;
    std::stack<std::shared_ptr<Function>> stack;
    stack.push(node);
    
    while (!stack.empty()) {
        std::shared_ptr<Function> current_node = stack.top();
        node_states[current_node] = VISITING;
        
        std::shared_ptr<Function> unvisited_child;
        bool all_child_node_visited = true;

        for (std::shared_ptr<Function>& fn : current_node->next_function) {
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


namespace autograd {

    void backward(Tensor& input_tensor, Tensor& input_grad) {
        input_grad.grad = std::shared_ptr<Tensor>(&input_grad);
        std::vector<std::shared_ptr<Function>> result_graph = topological_sort(input_grad.grad_fn);
        for (std::shared_ptr<Function>& fn : result_graph) {
            std::cout << fn << "\n";
        }
    }

}