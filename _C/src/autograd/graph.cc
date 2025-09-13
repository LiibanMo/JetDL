#include "jetdl/autograd/graph.h"

#include <memory>
#include <vector>

#include "jetdl/autograd.h"

namespace jetdl {
namespace autograd {

Graph::Graph(const Tensor& tensor) {
  if (!tensor.grad_fn) {
    return;
  }

  std::vector<std::shared_ptr<Function>> stack;
  std::vector<std::shared_ptr<Function>> visited;

  stack.push_back(tensor.grad_fn);
  visited.push_back(tensor.grad_fn);

  while (!stack.empty()) {
    auto current_fn = stack.back();
    stack.pop_back();

    fns_.push_back(current_fn);

    for (const auto& prev_tensor : current_fn->prev_tensors) {
      if (prev_tensor && prev_tensor->grad_fn) {
        auto next_fn = prev_tensor->grad_fn;
        bool already_visited = false;
        for (const auto& fn : visited) {
          if (fn == next_fn) {
            already_visited = true;
            break;
          }
        }

        if (!already_visited) {
          visited.push_back(next_fn);
          stack.push_back(next_fn);
        }
      }
    }
  }
}

void Graph::backward() {
  for (auto it = fns_.rbegin(); it != fns_.rend(); ++it) {
    auto& fn = *it;
    fn->apply(*fn->tensor.lock().get());
  }
}

}  // namespace autograd
}  // namespace jetdl
