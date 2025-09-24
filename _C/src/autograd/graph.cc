#include "jetdl/autograd/graph.h"

#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "jetdl/autograd.h"
#include "jetdl/math.h"
#include "jetdl/tensor.h"

namespace jetdl {

void Graph::traverse(std::shared_ptr<Tensor>& tensor) {
  this->graph.clear();
  if (!tensor->grad_fn) {
    return;
  }

  std::vector<std::shared_ptr<Function>> s1;
  std::unordered_set<std::shared_ptr<Function>> visited;

  s1.push_back(tensor->grad_fn);
  visited.insert(tensor->grad_fn);

  while (!s1.empty()) {
    const auto& fn = s1.back();
    s1.pop_back();
    this->graph.push_back(fn);

    for (const auto& next_fn : fn->next_functions) {
      if (next_fn && visited.find(next_fn) == visited.end()) {
        s1.push_back(next_fn);
        visited.insert(next_fn);
      }
    }
  }
}

void Graph::apply() {
  for (const auto& fn : this->graph) {
    std::shared_ptr<Tensor>& grad = fn->tensor.lock()->grad;
    if (grad == nullptr) {
      continue;
    }

    std::vector<std::shared_ptr<Tensor>> input_grads = fn->apply(grad);

    if (input_grads.size() != fn->saved_tensors.size()) {
      throw std::runtime_error(
          "INTERNAL: input_grads.size() != fn->saved_tensors.size()");
    }

    for (size_t i = 0; i < input_grads.size(); i++) {
      std::shared_ptr<Tensor>& tensor = fn->saved_tensors[i];
      std::shared_ptr<Tensor>& grad = input_grads[i];

      if (tensor->grad) {
        tensor->grad = math::add(tensor->grad, grad);
      } else {
        fn->saved_tensors[i]->grad = input_grads[i];
      }
    }
  }
}

}  // namespace jetdl
