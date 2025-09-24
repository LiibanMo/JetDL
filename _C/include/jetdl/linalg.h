#ifndef JETDL_LINALG_H
#define JETDL_LINALG_H

#include <memory>

#include "jetdl/tensor.h"

namespace jetdl {
namespace linalg {

std::shared_ptr<Tensor> dot(std::shared_ptr<Tensor>& a,
                            std::shared_ptr<Tensor>& b);

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor>& a,
                               std::shared_ptr<Tensor>& b);

std::shared_ptr<Tensor> T(std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> mT(std::shared_ptr<Tensor>& a);

}  // namespace linalg
}  // namespace jetdl

#endif
