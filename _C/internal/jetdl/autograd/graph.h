#ifndef JETDL_AUTOGRAD_GRAPH_HPP
#define JETDL_AUTOGRAD_GRAPH_HPP

#include "jetdl/tensor.h"
#include <memory>
#include <vector>

namespace jetdl {
namespace autograd {

class Graph {
 public:
  Graph(const Tensor& tensor);
  void backward();

 private:
  std::vector<std::shared_ptr<Function>> fns_;
};

}  // namespace autograd
}  // namespace jetdl

#endif
