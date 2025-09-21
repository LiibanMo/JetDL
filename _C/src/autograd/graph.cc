#include "jetdl/autograd/graph.h"

#include <memory>
#include <unordered_set>
#include <vector>

#include "jetdl/autograd.h"
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

void Graph::apply(std::shared_ptr<Tensor>& grad) {
  for (const auto& fn : this->graph) {
    const std::vector<std::shared_ptr<Tensor>>& grads = fn->apply(grad);
    if (grads.size() != fn->saved_tensors.size()) {
      throw std::runtime_error(
          "grad.size != fn->saved_tensor.size() in Graph::apply\n");
    }
    for (size_t i = 0; i < fn->saved_tensors.size(); i++) {
      std::shared_ptr<Tensor>& tensor = fn->saved_tensors[i];
      tensor->grad = grads[i];
    }
  }
}

}  // namespace jetdl
