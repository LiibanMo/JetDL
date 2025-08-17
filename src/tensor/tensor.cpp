#include "tensor.hpp"
#include "utils/metadata.hpp"

namespace py = pybind11;

Tensor::Tensor(py::list& data, bool requires_grad) {
    this->_data = utils::metadata::flatten_nested_pylist(data);
    utils::metadata::assign_basic_metadata(*this, utils::metadata::get_shape(data));

    this->requires_grad = requires_grad;
    this->grad_fn = nullptr;
    if (this->requires_grad) {
        this->grad = std::shared_ptr<Tensor>();
        utils::metadata::assign_basic_metadata(*this->grad, this->shape);
        this->grad->requires_grad = false;
    } else {
        this->grad = nullptr;
    }

    this->is_contiguous = true;
    this->is_leaf = true;
}

Tensor::Tensor(const float data, bool requires_grad) {
    this->_data = std::shared_ptr<float[]>(new float[1]);
    this->_data[0] = data;
    utils::metadata::assign_basic_metadata(*this, {});

    this->requires_grad = requires_grad;
    this->grad_fn = nullptr;
    this->grad = nullptr;

    this->is_contiguous = true;
    this->is_leaf = true;
}

Tensor::Tensor() {
    this->_data = nullptr;
    this->grad_fn = nullptr;
    this->grad = nullptr;

    this->is_contiguous = true;
    this->is_leaf = true;
};

Tensor Tensor::copy() {
    Tensor result_tensor = Tensor();

    result_tensor._data = std::make_shared<float[]>(this->size);
    std::copy(this->_data.get(), this->_data.get() + this->size, result_tensor._data.get());
    
    utils::metadata::assign_basic_metadata(result_tensor, this->shape);

    result_tensor.requires_grad = false;

    return result_tensor;
}   