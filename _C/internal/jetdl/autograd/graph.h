#ifndef JETDL_AUTOGRAD_GRAPH_HPP
#define JETDL_AUTOGRAD_GRAPH_HPP

#include <memory>
#include <mutex>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {

class Graph {
 private:
  // Functions grouped by level for parallel execution
  // Level 0 = root (output's grad_fn), higher levels depend on lower levels
  std::vector<std::vector<std::shared_ptr<Function>>> levels = {};

  // Mutex for thread-safe gradient accumulation
  std::mutex grad_mutex;

 public:
  Graph() = default;

  void traverse(std::shared_ptr<Tensor>& tensor);
  void apply();
};

}  // namespace jetdl

#endif
