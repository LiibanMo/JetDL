#include "ops.hpp"
#include "kernel.hpp"
#include "utils/auxiliary.hpp"
#include "utils/broadcast.hpp"
#include "utils/check.hpp"
#include "utils/metadata.hpp"

#include <memory>

#include <unordered_map>

using BinaryOperation = std::function<void(const float*, const float*, float*, const int)>;
using BinaryOperationScalars = std::function<void(const float, const float, float&)>;

std::unordered_map<std::string, BinaryOperation> registered_operations;
std::unordered_map<std::string, BinaryOperationScalars>  registered_operations_on_scalars;

void register_basic_ops() {
    registered_operations["ADD"] = [] (const float* a, const float* b, float* c, const int N) {
        c_add_cpu(a, b, c, N);
    };
    registered_operations["SUB"] = [] (const float* a, const float* b, float* c, const int N) {
        c_sub_cpu(a, b, c, N);
    };
    registered_operations["MUL"] = [] (const float* a, const float* b, float* c, const int N) {
        c_mul_cpu(a, b, c, N);
    };
    registered_operations["DIV"] = [] (const float* a, const float* b, float* c, const int N) {
        c_div_cpu(a, b, c, N);
    };
}

void register_basic_ops_on_scalars() {
    registered_operations_on_scalars["ADD"] = [] (const float a, const float b, float& c) {
        c = a + b;
    };
    registered_operations_on_scalars["SUB"] = [] (const float a, const float b, float& c) {
        c = a - b;
    };
    registered_operations_on_scalars["MUL"] = [] (const float a, const float b, float& c) {
        c = a * b;
    };
    registered_operations_on_scalars["DIV"] = [] (const float a, const float b, float& c) {
        c = a / b;
    };
}

Tensor c_ops(const Tensor& a, const Tensor& b, const std::string op) {
    utils::check::ops_broadcast_conditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();
    utils::broadcast::BroadcastingUtilsObject BroadcastUtils(a.shape, b.shape, false);

    // ----- Assigning metadata -----
    result_tensor.shape = BroadcastUtils.get_result_shape();
    result_tensor.ndim = utils::metadata::get_ndim(result_tensor.shape);
    result_tensor.size = utils::metadata::get_size(result_tensor.shape);
    result_tensor.strides = utils::metadata::get_strides(result_tensor.shape);
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    // ------------------------------

    const int MAX_NDIM = result_tensor.ndim;

    utils::IntPtrs strides = BroadcastUtils.get_broadcast_strides();

    std::unique_ptr<int[]> stridesA = std::move(strides.ptr1);
    std::unique_ptr<int[]> idxsA = utils::populate_linear_idxs(result_tensor.shape, stridesA.get(), 1);
    std::unique_ptr<int[]> stridesB = std::move(strides.ptr2);
    std::unique_ptr<int[]> idxsB = utils::populate_linear_idxs(result_tensor.shape, stridesB.get(), 1);

    const int NA = a.shape[a.ndim-1];
    const int NB = b.shape[b.ndim-1];
    const int N = (NA > NB) ? NA : NB;

    const int DATA_VEC_SIZE = utils::factor_ceiling_func(N, BLOCK_N_COLS);

    std::unique_ptr<float[]> result_vec = std::make_unique<float[]>(DATA_VEC_SIZE);
    std::unique_ptr<float[]> data1_vec = std::make_unique<float[]>(DATA_VEC_SIZE);
    std::unique_ptr<float[]> data2_vec = std::make_unique<float[]>(DATA_VEC_SIZE);

    const int TOTAL_NUM_ROWS = result_tensor.size / result_tensor.shape[MAX_NDIM-1];
    
    register_basic_ops();
    auto it = registered_operations.find(op);

    result_tensor._data = std::make_shared<float[]>(result_tensor.size);

    if (NA == NB) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::copy(a._data.get() + idxsA[row], a._data.get() + idxsA[row] + NA, data1_vec.get());
            std::copy(b._data.get() + idxsB[row], b._data.get() + idxsB[row] + NB, data2_vec.get());
            it->second(data1_vec.get(), data2_vec.get(), result_vec.get(), DATA_VEC_SIZE);
            std::copy(result_vec.get(), result_vec.get() + N, result_tensor._data.get() + row * N);
        }
    } else if (NA < NB && NA == 1) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::fill(data1_vec.get(), data1_vec.get() + N, a._data[idxsA[row]]);
            std::copy(b._data.get() + idxsB[row], b._data.get() + idxsB[row] + NB, data2_vec.get());
            it->second(data1_vec.get(), data2_vec.get(), result_vec.get(), DATA_VEC_SIZE);
            std::copy(result_vec.get(), result_vec.get() + N, result_tensor._data.get() + row * N);
        }
    } else if (NA > NB && NB == 1) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::copy(a._data.get() + idxsA[row], a._data.get() + idxsA[row] + NA, data1_vec.get());
            std::fill(data2_vec.get(), data2_vec.get() + N, b._data[idxsB[row]]);
            it->second(data1_vec.get(), data2_vec.get(), result_vec.get(), DATA_VEC_SIZE);
            std::copy(result_vec.get(), result_vec.get() + N, result_tensor._data.get() + row * N);
        }
    } 

    return result_tensor;
}

Tensor c_ops_scalar_a(const float a, const Tensor& b, const std::string op) {
    Tensor result_tensor = Tensor();

    // ----- Assigning metadata -----
    result_tensor.shape = b.shape;
    result_tensor.ndim = b.ndim;
    result_tensor.size = b.size;
    result_tensor.strides = b.strides;
    result_tensor.requires_grad = b.requires_grad;
    // ------------------------------

    const int N = b.shape[b.ndim-1];

    const int DATA_VEC_SIZE = utils::factor_ceiling_func(N, BLOCK_N_COLS);
    const int TOTAL_NUM_ROWS = result_tensor.size / N;

    std::unique_ptr<float[]> result_ptr = std::make_unique<float[]>(DATA_VEC_SIZE);
    std::unique_ptr<float[]> data_a_vec = std::make_unique<float[]>(DATA_VEC_SIZE);
    std::unique_ptr<float[]> data_b_vec = std::make_unique<float[]>(DATA_VEC_SIZE);

    std::fill(data_a_vec.get(), data_a_vec.get() + DATA_VEC_SIZE, a);

    register_basic_ops();
    auto it = registered_operations.find(op);

    result_tensor._data = std::make_shared<float[]>(result_tensor.size);

    for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
        std::copy(b._data.get() + row * N, b._data.get() + (row + 1) * N, data_b_vec.get());
        it->second(data_a_vec.get(), data_b_vec.get(), result_ptr.get(), DATA_VEC_SIZE);
        std::copy(result_ptr.get(), result_ptr.get() + N, result_tensor._data.get() + row * N);
    }

    return result_tensor;
}

Tensor c_ops_scalar_b(const Tensor& a, const float b, const std::string op) {
    Tensor result_tensor = Tensor();
    
    // ----- Assigning metadata -----
    result_tensor.shape = a.shape;
    result_tensor.ndim = a.ndim;
    result_tensor.size = a.size;
    result_tensor.strides = a.strides;
    result_tensor.requires_grad = a.requires_grad;
    // ------------------------------
    
    const int N = a.shape[a.ndim-1];

    const int DATA_VEC_SIZE = utils::factor_ceiling_func(N, BLOCK_N_COLS);
    const int TOTAL_NUM_ROWS = result_tensor.size / N;

    std::unique_ptr<float[]> result_ptr = std::make_unique<float[]>(DATA_VEC_SIZE);
    std::unique_ptr<float[]> data_a_vec = std::make_unique<float[]>(DATA_VEC_SIZE);
    std::unique_ptr<float[]> data_b_vec = std::make_unique<float[]>(DATA_VEC_SIZE);

    std::fill(data_b_vec.get(), data_b_vec.get() + N, b);

    register_basic_ops();
    auto it = registered_operations.find(op);

    result_tensor._data = std::make_shared<float[]>(result_tensor.size);

    for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
        std::copy(a._data.get() + row * N, a._data.get() + (row + 1) * N, data_a_vec.get());
        it->second(data_a_vec.get(), data_b_vec.get(), result_ptr.get(), DATA_VEC_SIZE);
        std::copy(result_ptr.get(), result_ptr.get() + N, result_tensor._data.get() + row * N);
    }

    return result_tensor;
}

Tensor c_ops_scalars(const Tensor& a, const Tensor& b, const std::string op) {
    Tensor result_tensor = Tensor();
    
    // ----- Assigning metadata -----
    result_tensor.shape = {};
    result_tensor.ndim = 0;
    result_tensor.size = 1;
    result_tensor.strides = {};
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    result_tensor.is_contiguous = true;
    result_tensor.is_leaf = false;
    // ------------------------------
    
    register_basic_ops_on_scalars();
    auto it = registered_operations_on_scalars.find(op);
    
    result_tensor._data = std::make_shared<float[]>(1);
    
    it->second(a._data[0], b._data[0], result_tensor._data[0]);

    return result_tensor;
}