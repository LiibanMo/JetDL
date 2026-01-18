#include "jetdl/autograd/graph.h"

#include <cstddef>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
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
  // Use raw pointers as keys for faster hash/comparison (objects stay alive
  // via shared_ptr in next_functions). Also maintain a vector of shared_ptrs
  // to keep ownership during traversal, and a map for O(1) index lookup.
  std::unordered_map<Function*, size_t> ptr_to_idx;
  std::vector<std::shared_ptr<Function>> all_fns_owned;
  std::vector<std::shared_ptr<Function>> stack;

  stack.push_back(tensor->grad_fn);

  while (!stack.empty()) {
    auto fn = std::move(stack.back());
    stack.pop_back();

    Function* fn_ptr = fn.get();
    if (ptr_to_idx.find(fn_ptr) != ptr_to_idx.end()) {
      continue;
    }
    ptr_to_idx[fn_ptr] = all_fns_owned.size();
    all_fns_owned.push_back(fn);

    for (const auto& next_fn : fn->next_functions) {
      if (next_fn) {
        stack.push_back(next_fn);
      }
    }
  }

  const size_t num_fns = all_fns_owned.size();

  // Step 2: Compute in-degree for each function using index-based arrays
  // in_degree[i] = number of functions that have all_fns_owned[i] in their
  // next_functions. Using vectors instead of maps for cache-friendly access.
  std::vector<size_t> in_degree(num_fns, 0);
  for (const auto& fn : all_fns_owned) {
    for (const auto& next_fn : fn->next_functions) {
      if (next_fn) {
        auto it = ptr_to_idx.find(next_fn.get());
        if (it != ptr_to_idx.end()) {
          in_degree[it->second]++;
        }
      }
    }
  }

  // Step 3: Compute levels using modified Kahn's algorithm
  // Level = longest path from any root (in_degree=0) to this function
  // Using vector for cache-friendly access
  std::vector<size_t> fn_levels(num_fns, 0);

  // Initialize: functions with in_degree 0 are at level 0
  std::queue<size_t> queue;
  for (size_t i = 0; i < num_fns; i++) {
    if (in_degree[i] == 0) {
      queue.push(i);
    }
  }

  // Process in topological order, computing max level
  while (!queue.empty()) {
    size_t idx = queue.front();
    queue.pop();
    const auto& fn = all_fns_owned[idx];
    size_t current_level = fn_levels[idx];

    for (const auto& next_fn : fn->next_functions) {
      if (next_fn) {
        auto it = ptr_to_idx.find(next_fn.get());
        if (it != ptr_to_idx.end()) {
          size_t next_idx = it->second;
          // Update level to max of all incoming paths
          fn_levels[next_idx] =
              std::max(fn_levels[next_idx], current_level + 1);

          in_degree[next_idx]--;
          if (in_degree[next_idx] == 0) {
            queue.push(next_idx);
          }
        }
      }
    }
  }

  // Step 4: Group functions by level
  size_t max_level = 0;
  for (size_t level : fn_levels) {
    max_level = std::max(max_level, level);
  }

  this->levels.resize(max_level + 1);
  for (size_t i = 0; i < num_fns; i++) {
    this->levels[fn_levels[i]].push_back(all_fns_owned[i]);
  }
}

void Graph::apply() {
  // Process each level sequentially, but functions within a level in parallel
  for (const auto& level_fns : this->levels) {
    const ptrdiff_t num_fns = static_cast<ptrdiff_t>(level_fns.size());

#ifdef JETDL_WITH_OPENMP
#pragma omp parallel for schedule(dynamic) if (num_fns > 1)
#endif
    for (ptrdiff_t idx = 0; idx < num_fns; idx++) {
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
