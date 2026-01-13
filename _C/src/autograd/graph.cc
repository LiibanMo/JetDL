#include "jetdl/autograd/graph.h"

#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef JETDL_WITH_OPENMP
#include <omp.h>
#endif

#include "jetdl/autograd.h"
#include "jetdl/math.h"
#include "jetdl/routines.h"
#include "jetdl/tensor.h"

namespace jetdl {

void Graph::traverse(std::shared_ptr<Tensor>& tensor) {
  this->levels.clear();
  if (!tensor->grad_fn) {
    return;
  }

  // Step 1: Collect all functions using DFS
  std::unordered_set<std::shared_ptr<Function>> all_fns;
  std::vector<std::shared_ptr<Function>> stack;

  stack.push_back(tensor->grad_fn);

  while (!stack.empty()) {
    auto fn = stack.back();
    stack.pop_back();

    if (all_fns.find(fn) != all_fns.end()) {
      continue;
    }
    all_fns.insert(fn);

    for (const auto& next_fn : fn->next_functions) {
      if (next_fn) {
        stack.push_back(next_fn);
      }
    }
  }

  // Step 2: Compute in-degree for each function
  // in_degree[fn] = number of functions that have fn in their next_functions
  std::unordered_map<std::shared_ptr<Function>, size_t> in_degree;
  for (const auto& fn : all_fns) {
    in_degree[fn] = 0;
  }
  for (const auto& fn : all_fns) {
    for (const auto& next_fn : fn->next_functions) {
      if (next_fn && all_fns.find(next_fn) != all_fns.end()) {
        in_degree[next_fn]++;
      }
    }
  }

  // Step 3: Compute levels using modified Kahn's algorithm
  // Level = longest path from any root (in_degree=0) to this function
  std::unordered_map<std::shared_ptr<Function>, size_t> fn_levels;

  // Initialize: functions with in_degree 0 are at level 0
  std::queue<std::shared_ptr<Function>> queue;
  for (const auto& fn : all_fns) {
    if (in_degree[fn] == 0) {
      fn_levels[fn] = 0;
      queue.push(fn);
    }
  }

  // Process in topological order, computing max level
  while (!queue.empty()) {
    auto fn = queue.front();
    queue.pop();

    for (const auto& next_fn : fn->next_functions) {
      if (next_fn && all_fns.find(next_fn) != all_fns.end()) {
        // Update level to max of all incoming paths
        size_t new_level = fn_levels[fn] + 1;
        if (fn_levels.find(next_fn) == fn_levels.end()) {
          fn_levels[next_fn] = new_level;
        } else {
          fn_levels[next_fn] = std::max(fn_levels[next_fn], new_level);
        }

        in_degree[next_fn]--;
        if (in_degree[next_fn] == 0) {
          queue.push(next_fn);
        }
      }
    }
  }

  // Step 4: Group functions by level
  size_t max_level = 0;
  for (const auto& [fn, level] : fn_levels) {
    max_level = std::max(max_level, level);
  }

  this->levels.resize(max_level + 1);
  for (const auto& [fn, level] : fn_levels) {
    this->levels[level].push_back(fn);
  }
}

void Graph::apply() {
  // Process each level sequentially, but functions within a level in parallel
  for (const auto& level_fns : this->levels) {
    const size_t num_fns = level_fns.size();

#ifdef JETDL_WITH_OPENMP
    #pragma omp parallel for schedule(dynamic) if(num_fns > 1)
#endif
    for (size_t idx = 0; idx < num_fns; idx++) {
      const auto& fn = level_fns[idx];

      std::shared_ptr<Tensor> output_tensor = fn->tensor.lock();
      if (!output_tensor || output_tensor->grad == nullptr) {
        continue;
      }

      std::shared_ptr<Tensor>& grad = output_tensor->grad;
      std::vector<std::shared_ptr<Tensor>> input_grads = fn->apply(grad);

      if (input_grads.size() != fn->saved_tensors.size()) {
        throw std::runtime_error(
            "INTERNAL: input_grads.size() != fn->saved_tensors.size()");
      }

      for (size_t i = 0; i < input_grads.size(); i++) {
        std::shared_ptr<Tensor>& tensor = fn->saved_tensors[i];
        std::shared_ptr<Tensor>& input_grad = input_grads[i];

        if (input_grad == nullptr) {
          continue;
        }

        // Thread-safe gradient accumulation
        {
#ifdef JETDL_WITH_OPENMP
          std::lock_guard<std::mutex> lock(this->grad_mutex);
#endif
          if (tensor->grad) {
            math::add_inplace(tensor->grad, input_grad);
          } else {
            tensor->grad = contiguous(input_grad);
          }
        }
      }
    }
  }
}

}  // namespace jetdl
