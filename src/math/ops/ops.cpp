#include "ops.hpp"
#include "kernel.hpp"
#include "utils/auxillary.hpp"
#include "utils/broadcast.hpp"
#include "utils/check.hpp"
#include "utils/metadata.hpp"

#include <cstdlib>
#include <functional>
#include <unordered_map>
#include <memory>

using BinaryOperation = std::function<void(const float*, const float*, float*, const int)>;

std::unordered_map<std::string, BinaryOperation> registered_operations;

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

Tensor c_ops(const Tensor& a, const Tensor& b, std::string op) {
    utils::check::opsBroadcastConditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();
    utils::broadcast::BroadcastingUtilsObject BroadcastUtils(a.shape, b.shape, false);

    // ----- Assigning metadata -----
    result_tensor.shape = BroadcastUtils.getResultShape();
    result_tensor.ndim = utils::metadata::getNumDim(result_tensor.shape);
    result_tensor.size = utils::metadata::getSize(result_tensor.shape);
    result_tensor.strides = utils::metadata::getStrides(result_tensor.shape);
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    // ------------------------------

    const int MAX_NDIM = result_tensor.ndim;

    utils::IntPtrs strides = BroadcastUtils.getBroadcastStrides();

    std::unique_ptr<int[]> stridesA = std::move(strides.ptr1);
    std::unique_ptr<int[]> idxsA = utils::populateLinearIdxs(result_tensor.shape, stridesA.get(), 1);
    std::unique_ptr<int[]> stridesB = std::move(strides.ptr2);
    std::unique_ptr<int[]> idxsB = utils::populateLinearIdxs(result_tensor.shape, stridesB.get(), 1);

    const int NA = a.shape[a.ndim-1];
    const int NB = b.shape[b.ndim-1];
    const int N = (NA > NB) ? NA : NB;

    const int DATA_VEC_SIZE = utils::factorCeilingFunc(N, BLOCK_N_COLS);

    std::unique_ptr<float[]> result_vec(new float[DATA_VEC_SIZE]());
    std::unique_ptr<float[]> data1_vec(new float[DATA_VEC_SIZE]());
    std::unique_ptr<float[]> data2_vec(new float[DATA_VEC_SIZE]());
    if (!result_vec || !data1_vec || !data2_vec) {
        throw std::runtime_error("Memory allocation failed.\n");
    }

    const int TOTAL_NUM_ROWS = result_tensor.size / result_tensor.shape[MAX_NDIM-1];
    result_tensor._data = std::shared_ptr<float[]>(new float[result_tensor.size]());
    
    register_basic_ops();
    auto it = registered_operations.find(op);

    if (NA == NB) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::memcpy(data1_vec.get(), a._data.get() + idxsA[row], NA * sizeof(float));
            std::memcpy(data2_vec.get(), b._data.get() + idxsB[row], NB * sizeof(float));
            it->second(data1_vec.get(), data2_vec.get(), result_vec.get(), DATA_VEC_SIZE);
            std::memcpy(result_tensor._data.get() + row * N, result_vec.get(), N * sizeof(float));
        }
    } else if (NA < NB && NA == 1) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::fill(data1_vec.get(), data1_vec.get() + N, a._data[idxsA[row]]);
            std::memcpy(data2_vec.get(), b._data.get() + idxsB[row], NB * sizeof(float));
            it->second(data1_vec.get(), data2_vec.get(), result_vec.get(), DATA_VEC_SIZE);
            std::memcpy(result_tensor._data.get() + row * N, result_vec.get(), N * sizeof(float));
        }
    } else if (NA > NB && NB == 1) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::memcpy(data1_vec.get(), a._data.get() + idxsA[row], NA * sizeof(float));
            std::fill(data2_vec.get(), data2_vec.get() + N, b._data[idxsB[row]]);
            it->second(data1_vec.get(), data2_vec.get(), result_vec.get(), DATA_VEC_SIZE);
            std::memcpy(result_tensor._data.get() + row * N, result_vec.get(), N * sizeof(float));
        }
    } else if (N == 1) {
        result_tensor._data[0] = a._data[0] + b._data[0];
    }

    return result_tensor;
}