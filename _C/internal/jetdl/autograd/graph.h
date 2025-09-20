#ifndef JETDL_AUTOGRAD_GRAPH_HPP
#define JETDL_AUTOGRAD_GRAPH_HPP

#include <memory>
#include <vector>

#include "jetdl/tensor.h"

namespace jetdl {

class Graph {
 private:
  std::vector<std::shared_ptr<Function>> graph = {};

 public:
  Graph() = default;

  void traverse(const Tensor& tensor);
  void apply(const std::shared_ptr<Tensor>& grad);
};

}  // namespace jetdl

#endif
