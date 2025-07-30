#include "ops.hpp"
#include "kernel.hpp"
#include "utils/auxillary.hpp"
#include "utils/broadcast.hpp"
#include "utils/check.hpp"
#include "utils/metadata.hpp"

#include <cstdlib>
#include <functional>
#include <unordered_map>

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

    int* stridesA = strides.ptr1;
    int* idxsA = utils::populateLinearIdxs(result_tensor.shape, stridesA, 1);
    free(stridesA);

    int* stridesB = strides.ptr2;
    int* idxsB = utils::populateLinearIdxs(result_tensor.shape, stridesB, 1);
    free(stridesB);

    const int NA = a.shape[a.ndim-1];
    const int NB = b.shape[b.ndim-1];
    const int N = (NA > NB) ? NA : NB;

    const int DATA_VEC_SIZE = utils::factorCeilingFunc(N, BLOCK_N_COLS);

    float* result_vec = (float*)std::calloc(DATA_VEC_SIZE, sizeof(float));
    float* data1_vec = (float*)std::calloc(DATA_VEC_SIZE, sizeof(float));
    float* data2_vec = (float*)std::calloc(DATA_VEC_SIZE, sizeof(float));
    if (!result_vec || !data1_vec || !data2_vec) {
        throw std::runtime_error("Memory allocation failed.\n");
    }

    const int TOTAL_NUM_ROWS = result_tensor.size / result_tensor.shape[MAX_NDIM-1];
    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);
    
    register_basic_ops();
    auto it = registered_operations.find(op);

    if (NA == NB) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::memcpy(&data1_vec[0], &a._data[idxsA[row]], NA * sizeof(float));
            std::memcpy(&data2_vec[0], &b._data[idxsB[row]], NB * sizeof(float));
            it->second(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
            std::memcpy(&result_tensor._data[row * N], &result_vec[0], N * sizeof(float));
        }
    } else if (NA < NB && NA == 1) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::fill(data1_vec, data1_vec + N, a._data[idxsA[row]]);
            std::memcpy(&data2_vec[0], &b._data[idxsB[row]], NB * sizeof(float));
            it->second(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
            std::memcpy(&result_tensor._data[row * N], &result_vec[0], N * sizeof(float));
        }
    } else if (NA > NB && NB == 1) {
        for (int row = 0; row < TOTAL_NUM_ROWS; row++) {
            std::memcpy(&data1_vec[0], &a._data[idxsA[row]], NA * sizeof(float));
            std::fill(data2_vec, data2_vec + N, b._data[idxsB[row]]);
            it->second(data1_vec, data2_vec, result_vec, DATA_VEC_SIZE);
            std::memcpy(&result_tensor._data[row * N], &result_vec[0], N * sizeof(float));
        }
    } else if (N == 1) {
        result_tensor._data[0] = a._data[0] + b._data[0];
    }

    free(idxsA);
    free(idxsB);
    free(result_vec);
    free(data1_vec);
    free(data2_vec);

    return result_tensor;
}