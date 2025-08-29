#include "routines.h"
#include "routines/creation/ones.h"
#include "tensor/tensor.h"

std::unique_ptr<Tensor, decltype(&destroy_tensor)> routines_ones(
    const std::vector<size_t> shape
) {
    Tensor* result_tensor = c_routines_ones(shape.data(), shape.size());
    return std::unique_ptr<Tensor, decltype(&destroy_tensor)>(
        result_tensor, &destroy_tensor
    );
}